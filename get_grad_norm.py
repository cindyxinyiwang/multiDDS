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
import copy

from fairseq import bleu
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
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
    src_dict = task.dictionary
    tgt_dict = task.label_dictionary
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )
    model = models[0]

    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    trainer = Trainer(args, task, model, criterion)
    epoch_itr, filtered_indices = trainer.get_train_iterator(epoch=0)
    # Update parameters every N batches
    update_freq = 1
    num_reset = 1
    datasize = -1

    for reset_idx in range(num_reset):
        print("resetting at step", reset_idx)
        # Initialize data iterator
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.epoch >= args.curriculum),
            offset=reset_idx*(args.update_language_sampling*args.update_freq[0]+1),
            datasize=datasize,
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch, no_progress_bar='simple',
        )

        for _, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
            for sample in samples:
                sample = trainer._prepare_sample(sample)
                grad_norm = task.get_grad_wrt_input(sample, model, criterion)
                #print(grad_norm)
                #print(grad_norm.size())
                for i, sample_id in enumerate(sample['id'].tolist()):
                    #target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                    target_tokens = sample['target'][i, :].int().cpu() + tgt_dict.nspecial
                    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], src_dict.pad())
                    src_str = src_dict.string(src_tokens[1:])
                    target_str = tgt_dict.string(target_tokens)
                    print('S-{}\t{}'.format(sample_id, src_str))
                    print('T-{}\t{}'.format(sample_id, target_str))
                    grad_norm_i = grad_norm[i, :].data.float().cpu().numpy()
                    #print(src_tokens)
                    #print(" ".join([str(g) for g in grad_norm_i]))
                    print('N-{}\t{}'.format(sample_id, " ".join([str(g) for g in grad_norm_i[1:len(src_tokens)-1]])))

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
