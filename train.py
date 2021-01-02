#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random

import numpy as np
import torch

from fairseq import bleu
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators, data_utils
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr, filtered_maxpos_indices = checkpoint_utils.load_checkpoint(args, trainer)

    # pretrain data actor
    # only the language actor model can be pretrained
    pretrain = True # temp bool for test purpose
    if pretrain and args.pretrain_data_actor and args.data_actor == 'ave':
        # pretrain the agent with LASER score
        # epoch_itr, indices = trainer.get_train_iterator(1)
        path = '/home/wtan12/multiDDS/'
        trainer.pretrain_LASER(path+'en-ps.laser-score', epoch_itr)
        # return

    compare_laser = False
    if compare_laser:
        epoch_itr, indices = trainer.get_train_iterator(1)
        print('Number of Indices: ', len(indices))
        scores = collections.defaultdict(float)
        # compare with laser label using R^2 Score, only used after model is trained
        # itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=False, shuffle=False)
        data_actor = trainer.data_actor
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=False,
            offset=0,
            datasize=-1,
        )
        for i, sample in enumerate(itr):
            sample = trainer._prepare_sample(sample)
            sample = list(sample.values())[0]
            score = data_actor(sample).cpu().detach().numpy().tolist()
            indices = sample['id'].data.cpu().numpy().ravel().tolist()
            for k, v in zip(indices, score):
                scores[k] = float(v[0])

        scores = sorted(scores.items(), key=lambda x: x[0])
        print('Number of Indices in Scoring file: ', len(scores))
        path = '/home/wtan12/multiDDS/'
        with open(path+'en-ps.laser-score', 'r') as r:
            data = r.read()
        laser_score = []
        for i, item in enumerate(data.split('\n')):
            laser_score.append(item)
        laser_score.pop()
        r2 = 0.0
        with open(path+'en-ps.dds_score', 'w') as f:
            for k, v in scores:
                f.write(str(v)+'\n')
                truth = float(laser_score[k])
                r2 += (truth-v)**2
        print('R2 Score compared to LASER file: ', r2)
        return



    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    if args.eval_bleu:
        generator = task.build_generator(args)
        args.maximize_best_checkpoint_metric = True
    else:
        generator = None
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        epoch_itr = train(args, trainer, task, epoch_itr, generator, filtered_maxpos_indices)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, generator)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)[0]
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr, generator=None, filtered_maxpos_indices=None):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf

    # data selection: reset epoch iter to filter out unselected data
    filter_data = epoch_itr.epoch % args.select_by_dds_epoch == 0
    if filter_data and args.select_by_dds_epoch > 0:
        epoch_itr, _ = trainer.get_filtered_train_iterator(epoch_itr.epoch, filtered_maxpos_indices=filtered_maxpos_indices)

    # if args.update_language_sampling > 0 and args.select_by_dds_epoch < 0 and (not args.data_actor_step_update):
    #     num_reset = len(epoch_itr.frozen_batches) // (args.update_language_sampling*args.update_freq[0]+1)
    #     datasize = args.update_language_sampling*args.update_freq[0]+1
    #     if num_reset * datasize < len(epoch_itr.frozen_batches):
    #         num_reset += 1
    # else:
    #     num_reset = 1
    #     datasize = -1
    # for reset_idx in range(num_reset):
    #     print("resetting at step", reset_idx)
        # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
        offset=0,
        datasize=-1,
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        #print(samples)

        # if args.extra_data_actor == 'ave_emb':
        #     update_actor = (i % args.extra_update_language_sampling == 0)
        # elif args.data_actor_step_update:
        #     update_actor = (i % args.update_language_sampling == 0)
        # elif args.data_actor == 'lan' and args.data_actor_step_update:
        #     update_actor = (i % args.update_language_sampling == 0)
        # else:
        #     update_actor = False
        # update sampling distribution
        # if args.update_language_sampling > 0 and i % args.update_language_sampling == 0 and args.data_actor != 'ave_emb' and not args.data_actor_step_update:
        #     if args.data_actor_multilin:
        #         trainer.update_language_sampler_multilin(args, epoch=epoch_itr.epoch)
        #     else:
        #         trainer.update_language_sampler(args)

        if ( epoch_itr.epoch > args.select_by_dds_epoch and args.select_by_dds_epoch > 0): update_actor = False
        update_actor=False
        log_output = trainer.train_step(samples, update_actor=update_actor)
        if log_output is None:
            continue

        # update the data selector
        if args.select_by_dds_epoch > 0 and i % args.update_data_selector == 0:
            trainer.update_data_selector(args)

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, generator)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()
    return epoch_itr


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets, generator=None):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    if args.eval_bleu:
        bleus = validate_translation(args, trainer, task, epoch_itr, generator)
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            noskip=True,
        )[0].next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())
        if args.eval_bleu:
            for k, v in bleus.items():
                extra_meters[k + ":bleu"].update(v)

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        if epoch_itr.epoch > args.switch_obj_epoch:
            for k, v in extra_meters.items():
                #print(k, v.avg)
                if k.endswith(":loss"):
                    k = k.split(":")[0]
                    trainer.valid_losses[k] = v.avg
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )

    if args.eval_bleu:
        return [sum(bleus.values())]
    else:
        return valid_losses

def validate_translation(args, trainer, task, epoch_itr, generator):
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    models = [trainer.get_model()]
    bleu_dict = {key: None for key in task.eval_lang_pairs}

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer_dict = {key: bleu.SacrebleuScorer() for key in task.eval_lang_pairs}
    else:
        scorer_dict = {key: bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk()) for key in task.eval_lang_pairs}

    itr = task.get_batch_iterator(
        dataset=task.dataset('valid'),
        max_tokens=args.max_tokens_valid,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
        noskip=True,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch,
        prefix='translate subset',
        no_progress_bar='simple'
    )

    num_sentences = 0
    has_target = True
    #with progress_bar.build_progress_bar(args, itr) as t:
    for samples in progress:
        if torch.cuda.is_available() and not args.cpu:
            samples = utils.move_to_cuda(samples)
        #if 'net_input' not in samples:
        #    continue

        prefix_tokens = None
        for key, sample in samples.items():
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                # Remove padding
                if args.sde:
                    src_tokens = target_tokens
                else:
                    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())

                # Either retrieve the original sentences or regenerate them from tokens.
                #if src_dict is not None:
                #    src_str = src_dict.string(src_tokens, args.remove_bpe)
                #else:
                #    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                #if not args.quiet:
                #    if src_dict is not None:
                #        print('S-{}\t{}'.format(sample_id, src_str))
                #    if has_target:
                #        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str="",
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=None,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                #if not args.quiet:
                #    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                #    print('P-{}\t{}'.format(
                #        sample_id,
                #        ' '.join(map(
                #            lambda x: '{:.4f}'.format(x),
                #            hypo['positional_scores'].tolist(),
                #        ))
                #    ))

                #    if args.print_alignment:
                #        print('A-{}\t{}'.format(
                #            sample_id,
                #            ' '.join(map(lambda x: str(utils.item(x)), alignment))
                #        ))

                # Score only the top hypothesis
                if has_target and j == 0:
                    if args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                    if hasattr(scorer_dict[key], 'add_string'):
                        scorer_dict[key].add_string(target_str, hypo_str)
                    else:
                        scorer_dict[key].add(target_tokens, hypo_tokens)

            num_sentences += sample['nsentences']
    for key, scorer in scorer_dict.items():
        bleu_dict[key] = scorer.score()
    return bleu_dict

def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
