# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import os

import numpy as np

from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
    MultiCorpusSampledDataset,
    TCSSampledDataset,
)

from . import FairseqTask, register_task


@register_task('multilingual_sentence_prediction')
class MultilingualSentencePredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='Truncate sequence to max_sequence_length')


        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--langs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('--eval-langs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('--dataset-type', default="round_robin", type=str,
                            help='[round_robin|multi|tcs]')

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self.label_dictionary = label_dictionary

        self.dataset_type = self.args.dataset_type
        self.langs = self.args.langs.split(",")
        self.eval_langs = self.args.eval_langs
        if self.eval_langs is None:
            self.eval_langs = self.langs

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        args.tokens_per_sample = args.max_positions

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, args.langs.split(',')[0], 'input0', 'dict.txt'),
            source=True,
        )
        print('| [input] dictionary: {} types'.format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, args.langs.split(',')[0], 'label', 'dict.txt'),
                source=False,
            )
            print('| [label] dictionary: {} types'.format(len(label_dict)))
        else:
            label_dict = data_dict
        return MultilingualSentencePredictionTask(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        if self.args.source_lang is not None:
            training = False
        else:
            training = True

        def get_path(type, lang, split):
            return os.path.join(self.args.data, lang, type, split)

        def make_dataset(type, lang, dictionary):
            split_path = get_path(type, lang, split)
            print(split_path)
            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        def lang_dataset(lang):
            input0 = make_dataset('input0',lang, self.source_dictionary)
            assert input0 is not None, 'could not find dataset: {}'.format(get_path('input0', lang, split))
            input1 = make_dataset('input1', lang, self.source_dictionary)
    
            if self.args.init_token is not None:
                input0 = PrependTokenDataset(input0, self.args.init_token)
    
            if input1 is None:
                src_tokens = input0
            else:
                if self.args.separator_token is not None:
                    input1 = PrependTokenDataset(input1, self.args.separator_token)
    
                src_tokens = ConcatSentencesDataset(input0, input1)
    
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_tokens))
    
            if self.args.truncate_sequence:
                src_tokens = TruncateDataset(src_tokens, self.args.max_positions)
    
            dataset = {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': RightPadDataset(
                        src_tokens,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    'src_lengths': NumelDataset(src_tokens, reduce=False),
                },
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(src_tokens, reduce=True),
            }
    
            if not self.args.regression_target:
                label_dataset = make_dataset('label', lang, self.target_dictionary)
                if label_dataset is not None:
                    dataset.update(
                        target=OffsetTokensDataset(
                            StripTokenDataset(
                                label_dataset,
                                id_to_strip=self.target_dictionary.eos(),
                            ),
                            offset=-self.target_dictionary.nspecial,
                        )
                    )
            else:
                label_path = "{0}.label".format(get_path('label', lang, split))
                if os.path.exists(label_path):
                    dataset.update(
                        target=RawLabelDataset([
                            float(x.strip()) for x in open(label_path).readlines()
                        ])
                    )
    
            nested_dataset = NestedDictionaryDataset(
                dataset,
                sizes=[src_tokens.sizes],
            )
    
            if self.args.no_shuffle:
                dataset = nested_dataset
            else:
                dataset = SortDataset(
                    nested_dataset,
                    # shuffle
                    sort_order=[shuffle],
                )
    
            print("| Loaded {0} with #samples: {1}".format(split, len(dataset)))
            return dataset
        if self.dataset_type == 'round_robin' or split != 'train':
            source_lang = self.args.source_lang
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([
                    (lang, lang_dataset(lang))
                    for lang in self.langs
                ]),
                eval_key=None if training else "%s" % (source_lang),
            )
        elif self.dataset_type == 'multi':
            self.datasets[split] =  MultiCorpusSampledDataset(
                OrderedDict([
                    (lang, lang_dataset(lang))
                    for lang in self.langs
                ]),
                sample_instance=self.args.sample_instance,
                split=split,
                datasize_t=self.args.datasize_t,
                alpha_p=self.args.alpha_p,
            )
        else:
            print('Error: dataset type unsupported')
            exit(0)
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_classification_head(
            'sentence_classification_head',
            num_classes=self.args.num_classes,
        )

        return model

    def max_positions(self):
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.label_dictionary

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False, data_actor=None, loss_copy=None, data_actor_out=None):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
        if (self.args.data_actor == 'ave_emb' or self.args.extra_data_actor == 'ave_emb') and data_actor is not None:
            data_score, sum_score, example_size = {}, 0, 0
            for lang_pair in self.model_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                cur_sample = sample[lang_pair]
                score = data_actor(cur_sample['net_input']['src_tokens'], cur_sample['target'])
                data_actor_out[lang_pair] = score
                data_score[lang_pair] = score
                sum_score += score.sum()
                example_size += cur_sample['nsentences']
            # normalize scores
            for lang_pair in self.model_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                #if self.args.out_score_type == 'exp':
                #    data_actor_out[lang_pair] = data_actor_out[lang_pair]/sum_score
                data_score[lang_pair] = data_score[lang_pair]*example_size/sum_score
                #print(data_score[lang_pair])
        else:
            data_score = None
        #print(sample)
        for lang in self.langs:
            if lang not in sample or sample[lang] is None or len(sample[lang]) == 0:
                continue
            if data_score is not None:
                score = data_score[lang]
            else:
                score = None
            loss, sample_size, logging_output, nll_loss_data = criterion(model, sample[lang], data_score=score, loss_copy=(loss_copy is not None))
            if loss_copy is not None:
                loss_copy[lang] = nll_loss_data
            if ignore_grad:
                loss *= 0
            else:
                optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[lang] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion, logging_output_keys=None):
        logging_output_keys = logging_output_keys or self.langs
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            key: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(key, {}) for logging_output in logging_outputs
            ])
            for key in logging_output_keys
        }
        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        #if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
        #    flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang in self.eval_langs:
                if lang not in sample or sample[lang] is None or len(sample[lang]) == 0:
                    continue
                loss, sample_size, logging_output, _ = criterion(model, sample[lang])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output
