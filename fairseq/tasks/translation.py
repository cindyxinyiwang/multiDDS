# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import torch
import numpy as np
import scipy
import copy
from fairseq.data.legacy.masked_lm_dataset import MaskedLMDataset
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.data import Dictionary

from fairseq import options, utils, checkpoint_utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
)

from . import FairseqTask, register_task


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
    src_tag=None, tgt_tag=None, src_tau=-1, tgt_tau=-1, epoch=0, id_to_sample_probabilities=None, lm=None,
    idx_to_src_gradnorm=None 
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
        id_to_sample_probabilities=id_to_sample_probabilities,
        lm=lm,
        idx_to_src_gradnorm=idx_to_src_gradnorm,
    )


@register_task('translation')
class TranslationTask(FairseqTask):
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
        parser.add_argument('--main-src-wordfreq', default=None, type=str,
                            help='word frequency file of the main train source')
        parser.add_argument('--dialect-src-wordfreq', default=None, type=str,
                            help='word frequency file of the dialect train source')
        parser.add_argument('--dialect-tau', default=1., type=float)
        parser.add_argument('--src-gradnorm-tau', default=1., type=float)
        parser.add_argument('--lm-path', default=None, type=str)
        parser.add_argument('--lm-dict-path', default=None, type=str)
        parser.add_argument('--lm-topk', default=0, type=int)
        parser.add_argument('--src-gradnorm-path', default=None, type=str)
        parser.add_argument('--src-gradnorm-nosoftmax', action='store_true')
        parser.add_argument('--exclude-self', action='store_true')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.src_gradnorm_path is not None:
            self.idx_to_src_gradnorm = {}
            with open(args.src_gradnorm_path, 'r') as myfile:
                for line in myfile:
                    if line.startswith("N-"):
                        toks = line.split()
                        id = int(toks[0].split('-')[1])
                        if id%2 == 0:
                            id = int(id / 2)
                            assert id not in self.idx_to_src_gradnorm
                            if self.args.src_gradnorm_nosoftmax:
                                array = np.array([float(t) for t in toks[1:]])
                                self.idx_to_src_gradnorm[id] = (array / np.sum(array)).tolist()
                            else:
                                self.idx_to_src_gradnorm[id] = scipy.special.softmax([float(t)*args.src_gradnorm_tau for t in toks[1:]]).tolist()
        else:
            self.idx_to_src_gradnorm = None

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
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models

        # build and load LM
        if args.lm_path is not None:
            #self.dictionary = MaskedLMDictionary.load(args.lm_dict_path)
            #lm_args = copy.deepcopy(args)
            #lm_args.arch = "xlm_base"

            #lm_args.max_tokens = 2048
            #lm_args.tokens_per_sample = 256
            #lm_args.num_segment = 5
           
            #lm_args.encoder_embed_dim = 256
            #lm_args.encoder_ffn_embed_dim = 512
            #lm_args.encoder_layers = 6
            #lm_args.encoder_attention_heads = 4
            #lm_args.bias_kv = False
            #lm_args.zero_attn = False
            #lm_args.encoder_embed_dim = 256
            #lm_args.share_encoder_input_output_embed = True
            #lm_args.encoder_learned_pos = True
            #lm_args.no_token_positional_embeddings = False
            #lm_args.activation_fn = 'gelu'
            #lm_args.encoder_normalize_before = False
            #lm_args.pooler_activation_fn = 'tanh'
            #lm_args.apply_bert_init = True
 
            #self.mlm = models.build_model(lm_args, self)

            #state = checkpoint_utils.load_checkpoint_to_cpu(args.lm_path)
            #self.mlm.dictionary = self.dictionary
            #self.mlm.dialect_tau = self.args.dialect_tau
            #self.mlm.topk = self.args.lm_topk
            ## verify dictionary 
            #for i in range(10):
            #    print(self.src_dict[i], self.dictionary[i])
            ## load model parameters
            #try:
            #    self.mlm.load_state_dict(state['model'], strict=True)
            #except Exception:
            #    raise Exception(
            #        'Cannot load model parameters from checkpoint {}; '
            #        'please ensure that the architectures match.'.format(args.lm_path)
            #    )
            #if self.cuda:
            #    print("move lm to cuda") 
            #    self.mlm = self.mlm.cuda()
            #self.mlm.eval()


            lm_args = copy.deepcopy(args)
            lm_args.arch = "roberta_base"
            lm_args.sample_break_mode = "complete"
            lm_args.tokens_per_sample = 512

            lm_args.encoder_embed_dim = 768
            lm_args.encoder_ffn_embed_dim = 3072
            lm_args.encoder_layers = 12
            lm_args.encoder_attention_heads = 12
            lm_args.encoder_learned_pos = True
            lm_args.no_token_positional_embeddings = False
            lm_args.activation_fn = 'gelu'
            lm_args.pooler_activation_fn = 'tanh'
 
            self.dictionary = Dictionary.load(args.lm_dict_path)
            mask_idx = self.dictionary.add_symbol('<mask>')
            from fairseq.models.roberta import RobertaModel
            self.mlm = RobertaModel.build_model_with_dict(lm_args, self.dictionary)

            state = checkpoint_utils.load_checkpoint_to_cpu(args.lm_path)
            ## verify dictionary 
            for i in range(10):
                print(self.src_dict[i], self.dictionary[i])

            # load model parameters
            try:
                self.mlm.load_state_dict(state['model'], strict=True)
                #from fairseq.models.roberta import RobertaModel
                #self.mlm = RobertaModel.from_pretrained(args.lm_path, checkpoint_file="checkpoint_best.pt")

            except Exception:
                raise Exception(
                    'Cannot load model parameters from checkpoint {}; '
                    'please ensure that the architectures match.'.format(args.lm_path)
                )
            if self.cuda:
                print("move lm to cuda") 
                self.mlm = self.mlm.cuda()
            self.mlm.eval()

            self.mlm.dictionary = self.dictionary
            self.mlm.mask_id = mask_idx
            self.mlm.dialect_tau = self.args.dialect_tau
            self.mlm.topk = self.args.lm_topk
            self.mlm.exclude_self = self.args.exclude_self
        else:
            self.mlm = None

        return models.build_model(args, self)


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

        if self.args.main_src_wordfreq is not None and self.args.dialect_src_wordfreq is not None:
            def word_idx_from_file(filename):
                idx = []
                with open(filename, 'r') as myfile:
                    for line in myfile:
                        idx.append(self.src_dict.index(line.split()[0]))
                return idx
            self.main_src_word_idx = word_idx_from_file(self.args.main_src_wordfreq) 
            self.dialect_src_word_idx = word_idx_from_file(self.args.dialect_src_wordfreq)
            idx_to_sample_prob = []
            for i, src_word in enumerate(self.main_src_word_idx):
                if self.args.dialect_tau == -1:
                    dialect_word_probs = np.array([1. for k in range(len(self.dialect_src_word_idx))])
                else:
                    dialect_word_probs = np.array([-np.absolute(k-i) for k in range(len(self.dialect_src_word_idx))])
                idx_to_sample_prob.append(dialect_word_probs)
            #self.idx_to_sample_prob = scipy.special.softmax(np.array(self.idx_to_sample_prob)*0.01, axis=1)
            idx_to_sample_prob = scipy.special.softmax(np.array(idx_to_sample_prob)*self.args.dialect_tau, axis=1)
            print(idx_to_sample_prob)
            self.idx_to_sample_prob = {}
            for i, src_word in enumerate(self.main_src_word_idx):
                self.idx_to_sample_prob[src_word] = idx_to_sample_prob[i]
            pass_item = (self.idx_to_sample_prob, self.dialect_src_word_idx)
        else:
            pass_item = None
        if split != 'train':
            src_tau = -1 
            tgt_tau = -1
            mlm = None
            idx_to_src_gradnorm = None
        else: 
            src_tau = self.args.source_tau 
            tgt_tau = self.args.target_tau 
            mlm = self.mlm
            idx_to_src_gradnorm = self.idx_to_src_gradnorm

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            src_tag=self.args.src_tag, tgt_tag=self.args.tgt_tag,
            src_tau=src_tau, tgt_tau=tgt_tau,
            epoch=epoch,
            id_to_sample_probabilities=pass_item,
            lm=mlm,
            idx_to_src_gradnorm=idx_to_src_gradnorm,
        )

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
        #if type(list(sample.values())[0]) == dict:
        #    for k, v in sample.items():
        #        loss, sample_size, logging_output, loss_data, dev_grad_dotprod = criterion(model, v, data_score=data_score, loss_copy=((data_score is not None) or loss_copy))
        #else:
        #    loss, sample_size, logging_output, loss_data, dev_grad_dotprod = criterion(model, sample, data_score=data_score, loss_copy=((data_score is not None) or loss_copy))
        loss, sample_size, logging_output, loss_data, dev_grad_dotprod = criterion(model, sample, data_score=data_score, loss_copy=((data_score is not None) or loss_copy))
        if ignore_grad:
            loss *= 0
        else:
            optimizer.backward(loss)
        return loss, sample_size, logging_output, loss_data, dev_grad_dotprod


