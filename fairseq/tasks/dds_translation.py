# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import torch
import numpy as np

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    BacktranslationDataset,
)

from fairseq.sequence_generator import SequenceGenerator
from . import FairseqTask, register_task


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
    src_tag=None, tgt_tag=None, src_tau=-1, tgt_tau=-1, epoch=0, 
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(
            data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        )
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        src_tag=src_tag,
        tgt_tag=tgt_tag,
        src_tau=src_tau,
        tgt_tau=tgt_tau,
    )

# ported from UnsupervisedMT
def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                             # to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                             # iterations, then will linearly increase to 1 until iteration 2000
    """
    split = x.split(',')
    if len(split) == 1:
        return float(x), None
    else:
        split = [s.split(':') for s in split]
        assert all(len(s) == 2 for s in split)
        assert all(k.isdigit() for k, _ in split)
        assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
        return float(split[0][1]), [(int(k), float(v)) for k, v in split]

@register_task('dds_translation')
class ddsTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on
        parser.add_argument('--bt-max-len-a', default=1.1, type=float, metavar='N',
                            help='generate back-translated sequences of maximum length ax + b, where x is the '
                                 'source length')
        parser.add_argument('--bt-max-len-b', default=10.0, type=float, metavar='N',
                            help='generate back-translated sequences of maximum length ax + b, where x is the '
                                 'source length')
        parser.add_argument('--bt-beam-size', default=1, type=int, metavar='N',
                            help='beam size used in beam search of online back-translation')
        parser.add_argument('--lambda-dds-config', default="0.0", type=str, metavar='CONFIG',
                            help='Cross-entropy reconstruction coefficient (denoising autoencoding)'
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.num_updates = 0
        self.lambda_dds, self.lambda_dds_steps = parse_lambda_config(args.lambda_dds_config)
        self.cuda = torch.cuda.is_available() and not args.cpu

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)), sde=args.sde)
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        #if self.args.bt_dds:
        #    # set up data actor finetune optimizer
        #    bt_params = []
        #    for lang_pair in self.lang_pairs:
        #        bt_lang_pair = _get_dds_bt_key(lang_pair)
        #        for p in model.models[bt_lang_pair].parameters():
        #            if p.requires_grad: bt_params.append(p)
        #    if self.args.bt_optimizer == "SGD":
        #        self.data_optimizer = torch.optim.SGD(bt_params, lr=self.args.data_actor_lr[0], momentum=self.args.bt_optimizer_momentum, nesterov=self.args.bt_optimizer_nesterov)
        #    elif self.args.bt_optimizer == "ASGD":
        #        self.data_optimizer = torch.optim.ASGD(bt_params, lr=self.args.data_actor_lr[0], t0=0)
        #    elif self.args.bt_optimizer == "Adam":
        #        self.data_optimizer = torch.optim.Adam(bt_params, lr=self.args.data_actor_lr[0])
        #if self.args.swa:
        #    self.data_optimizer = torchcontrib.optim.SWA(self.data_optimizer, swa_start=self.args.swa_start, swa_freq=self.args.swa_freq, swa_lr=self.args.swa_lr)
        #if self.args.swa_schedule_gamma is not None:
        #    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.data_optimizer, step_size=1, gamma=self.args.swa_schedule_gamma)
        # create SequenceGenerator for each model that has backtranslation dependency on it
        self.sequence_generator = SequenceGenerator(
            tgt_dict=self.tgt_dict,
            beam_size=args.bt_beam_size,
            max_len_a=args.bt_max_len_a,
            max_len_b=args.bt_max_len_b,
            sampling=self.args.sampling,
            sampling_topk=self.args.sampling_topk,
            temperature=self.args.temperature,
        )

        def backtranslate_fn(
            sample, model=model,
            sequence_generator=self.sequence_generator,
        ):
            return sequence_generator.generate(
                [model],
                sample,
            )
        self.backtranslator = backtranslate_fn
        return model

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        if not hasattr(self.args, "source_tau"): self.args.source_tau = -1
        if not hasattr(self.args, "target_tau"): self.args.target_tau = -1

        if not hasattr(self.args, 'source_tau'): self.args.source_tau = -1
        if not hasattr(self.args, 'target_tau'): self.args.target_tau = -1

        langpair_dataset = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            src_tag=self.args.src_tag, tgt_tag=self.args.tgt_tag,
            src_tau=self.args.source_tau, tgt_tau=self.args.target_tau,
            epoch=epoch,
        )
        if split == 'train':
            self.datasets[split] = langpair_dataset
            self.datasets[split] = BacktranslationDataset(
                tgt_dataset=langpair_dataset,
                backtranslation_fn=self.backtranslator,
                src_dict=self.src_dict, tgt_dict=self.tgt_dict,
                output_collater=langpair_dataset.collater,
                bt_langpair=True,
            )

        else:
            self.datasets[split] = langpair_dataset
        if split == 'valid':
            if self.args.max_tokens_valid is not None:
                max_tokens_valid = self.args.max_tokens_valid
            else:
                max_tokens_valid = None
            if self.args.max_sentences_valid is not None:
                max_sentences_valid = self.args.max_sentences_valid
            else:
                max_sentences_valid = None
            self.dev_itr = self.get_batch_iterator(
                dataset=self.dataset('valid'),
                max_tokens=max_tokens_valid,
                max_sentences=max_sentences_valid,
                max_positions=utils.resolve_max_positions(
                    self.max_positions(),
                ),
                ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=self.args.required_batch_size_multiple,
                seed=self.args.seed,
                num_shards=self.args.distributed_world_size,
                shard_id=self.args.distributed_rank,
                num_workers=self.args.num_workers,
                noskip=True,
            )[0]
            self.dev_itr.next_epoch_itr(shuffle=True)


    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary, src_tag=self.args.src_tag, tgt_tag=self.args.tgt_tag)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False, data_score=None, loss_copy=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        # bt standard loss
        if self.lambda_dds > 0:
            params = []
            for p in model.parameters():
                if p.requires_grad: 
                    params.append(p)
            params = params[1:]
            if self.dev_itr.end_of_epoch():
                    self.dev_itr.next_epoch_itr(shuffle=True)
            for valid_sample in self.dev_itr._cur_epoch_itr:
                valid_sample = utils.move_to_cuda(valid_sample) if self.cuda else valid_sample
                loss, v_sample_size, _, valid_nll_loss, _ = criterion(model, valid_sample, loss_copy=True)
                valid_nll_loss = valid_nll_loss.sum() / v_sample_size
                with torch.no_grad():
                    valid_grad = torch.autograd.grad(valid_nll_loss, params, only_inputs=True)
                break
            
            dds_loss, dds_sample_size, logging_output, loss_data, _ = criterion(model, sample[1], data_score=None, loss_copy=True)
            # B X T
            #nll_loss = nll_loss_data.sum(dim=1) / sample_size
            nll_loss = loss_data / dds_sample_size
            z = torch.ones_like(nll_loss).requires_grad_(True)
            #norm_z = torch.ones_like(nll_loss).requires_grad_(True)
            pgrad = torch.autograd.grad(nll_loss, params, grad_outputs=z, create_graph=True, retain_graph=True, only_inputs=True)
            #norm_pgrad = torch.autograd.grad(nll_loss, params, grad_outputs=norm_z, create_graph=True, retain_graph=True, only_inputs=True)
            # calculate jvp loss
            jvp_loss = 0
            norm_loss = 0
            vg_norm, pg_norm = 0, 0
            for vg, pg in zip(valid_grad, pgrad):
                jvp_loss += (vg.data*pg).sum()
            #for vg, pg in zip(valid_grad, norm_pgrad):
            #    vg_norm += vg.data.norm(2)**2
            #    pg_norm += pg.norm(2)**2
            #    #vg_norm += vg.data.norm(1)
            #    #pg_norm += pg.data.norm(1)
            #norm_loss = (vg_norm * pg_norm) ** 0.5
            with torch.no_grad():
                dev_grad_dotprod = torch.autograd.grad(jvp_loss, z, retain_graph=False, only_inputs=True)[0]
            dds_loss, dds_sample_size, logging_output, loss_data, _ = criterion(model, sample[1], data_score=dev_grad_dotprod, loss_copy=False)
            
            # gold loss
            gold_loss, sample_size, logging_output, loss_data, dev_grad_dotprod = criterion(model, sample[2], loss_copy=True)
            loss = dds_loss * self.lambda_dds + gold_loss * (1. - self.lambda_dds)
        else:
            # gold standard loss
            loss, sample_size, logging_output, loss_data, dev_grad_dotprod = criterion(model, sample[2], data_score=data_score, loss_copy=((data_score is not None) or loss_copy))
        if ignore_grad:
            loss *= 0
        else:
            optimizer.backward(loss)
        return loss, sample_size, logging_output, loss_data, dev_grad_dotprod

    def update_step(self, num_updates):
        def lambda_step_func(config, n_iter):
            """
            Update a lambda value according to its schedule configuration.
            """
            ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
            if len(ranges) == 0:
                assert n_iter >= config[-1][0]
                return config[-1][1]
            assert len(ranges) == 1
            i = ranges[0]
            x_a, y_a = config[i]
            x_b, y_b = config[i + 1]
            return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)

        if self.lambda_dds_steps is not None:
            self.lambda_dds = lambda_step_func(self.lambda_dds_steps, num_updates)
        self.num_updates = num_updates
