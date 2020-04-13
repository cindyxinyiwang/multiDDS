# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq import utils

from . import FairseqDataset
from fairseq.data.noising import UnsupervisedMTNoising 

def backtranslate_samples(samples, collate_fn, generate_fn, cuda=True, noising=None, print_idx=None):
    """Backtranslate a list of samples.

    Given an input (*samples*) of the form:

        [{'id': 1, 'source': 'hallo welt'}]

    this will return:

        [{'id': 1, 'source': 'hello world', 'target': 'hallo welt'}]

    Args:
        samples (List[dict]): samples to backtranslate. Individual samples are
            expected to have a 'source' key, which will become the 'target'
            after backtranslation.
        collate_fn (callable): function to collate samples into a mini-batch
        generate_fn (callable): function to generate backtranslations
        cuda (bool): use GPU for generation (default: ``True``)

    Returns:
        List[dict]: an updated list of samples with a backtranslated source
    """
    collated_samples = collate_fn(samples)
    s = utils.move_to_cuda(collated_samples) if cuda else collated_samples

    generated_sources = generate_fn(s)

    id_to_src = {
        sample['id']: sample['source'] for sample in samples
    }

    # Go through each tgt sentence in batch and its corresponding best
    # generated hypothesis and create a backtranslation data pair
    # {id: id, source: generated backtranslation, target: original tgt}
    #return samples

    #ret_samples =  [
    #    {'id': id.item(), 'target': id_to_src[id.item()], 'source': hypos[0]['tokens'].cpu()}
    #    for id, hypos in zip(collated_samples['id'], generated_sources)
    #]
    ret_samples = []
    print_samples = []
    for id, hypos in zip(collated_samples['id'], generated_sources):
        ret_samples.append({'id': id.item(), 'target': id_to_src[id.item()], 'source': hypos[0]['tokens'].cpu()})
        if print_idx is not None and id.item() == print_idx:
            print_samples.append(ret_samples[-1])
    if noising is not None:
        backward_samples = []
        for id, hypos in zip(collated_samples['id'], generated_sources):
            s = id_to_src[id.item()]
            src_len = torch.LongTensor([s.size(0)])
            s = s.unsqueeze(1)

            ns = noising.noising(s, src_len)
            ns = torch.t(ns)[0]

            backward_samples.append({'id': id.item(), 'source': ns, 'target': hypos[0]['tokens'].cpu()})
    else:
        backward_samples = [
            {'id': id.item(), 'source': id_to_src[id.item()], 'target': hypos[0]['tokens'].cpu()}
            for id, hypos in zip(collated_samples['id'], generated_sources)
        ]
    return ret_samples, backward_samples, print_samples



class BacktranslationDataset(FairseqDataset):
    """
    Sets up a backtranslation dataset which takes a tgt batch, generates
    a src using a tgt-src backtranslation function (*backtranslation_fn*),
    and returns the corresponding `{generated src, input tgt}` batch.

    Args:
        tgt_dataset (~fairseq.data.FairseqDataset): the dataset to be
            backtranslated. Only the source side of this dataset will be used.
            After backtranslation, the source sentences in this dataset will be
            returned as the targets.
        src_dict (~fairseq.data.Dictionary): the dictionary of backtranslated
            sentences.
        tgt_dict (~fairseq.data.Dictionary, optional): the dictionary of
            sentences to be backtranslated.
        backtranslation_fn (callable, optional): function to call to generate
            backtranslations. This is typically the `generate` method of a
            :class:`~fairseq.sequence_generator.SequenceGenerator` object.
            Pass in None when it is not available at initialization time, and
            use set_backtranslation_fn function to set it when available.
        output_collater (callable, optional): function to call on the
            backtranslated samples to create the final batch
            (default: ``tgt_dataset.collater``).
        cuda: use GPU for generation
    """

    def __init__(
        self,
        tgt_dataset,
        src_dict,
        tgt_dict=None,
        backtranslation_fn=None,
        output_collater=None,
        backward_output_collater=None,
        cuda=True,
        noising=False,
        bt_langpair=False,
        **kwargs
    ):
        self.tgt_dataset = tgt_dataset
        self.bt_langpair = bt_langpair 
        self.backtranslation_fn = backtranslation_fn
        self.output_collater = output_collater if output_collater is not None \
            else tgt_dataset.collater
        self.backward_output_collater = backward_output_collater if backward_output_collater is not None \
            else self.output_collater
        self.cuda = cuda if torch.cuda.is_available() else False
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        if noising:
            self.noising = UnsupervisedMTNoising(self.src_dict, max_word_shuffle_distance=0, word_dropout_prob=0.1, word_blanking_prob=0.1, bpe_cont_marker="▁")
            #self.noising = UnsupervisedMTNoising(self.src_dict, max_word_shuffle_distance=5, word_dropout_prob=0.2, word_blanking_prob=0.2, bpe_cont_marker="▁")
        else:
            self.noising = None

    def __getitem__(self, index):
        """
        Returns a single sample from *tgt_dataset*. Note that backtranslation is
        not applied in this step; use :func:`collater` instead to backtranslate
        a batch of samples.
        """
        return self.tgt_dataset[index]

    def __len__(self):
        return len(self.tgt_dataset)

    def set_backtranslation_fn(self, backtranslation_fn):
        self.backtranslation_fn = backtranslation_fn

    def collater(self, samples):
        """Merge and backtranslate a list of samples to form a mini-batch.

        Using the samples from *tgt_dataset*, load a collated target sample to
        feed to the backtranslation model. Then take the backtranslation with
        the best score as the source and the original input as the target.

        Note: we expect *tgt_dataset* to provide a function `collater()` that
        will collate samples into the format expected by *backtranslation_fn*.
        After backtranslation, we will feed the new list of samples (i.e., the
        `(backtranslated source, original source)` pairs) to *output_collater*
        and return the result.

        Args:
            samples (List[dict]): samples to backtranslate and collate

        Returns:
            dict: a mini-batch with keys coming from *output_collater*
        """
        if self.bt_langpair:
            if samples[0].get('is_dummy', False):
                return samples
            print("gold standard src-trg")
            gold_standard_samples = samples[:]
            src_str = self.src_dict.string(gold_standard_samples[0]['source'])
            tgt_str = self.tgt_dict.string(gold_standard_samples[0]['target'])
            print_idx = gold_standard_samples[0]['id']
            print(src_str)
            print(tgt_str)
            #return samples
            samples, backward_samples, print_samples = backtranslate_samples(
                samples=samples,
                collate_fn=self.tgt_dataset.collater,
                generate_fn=(
                    lambda net_input: self.backtranslation_fn(net_input)
                ),
                cuda=self.cuda,
                noising=self.noising,
                print_idx=print_idx,
            )
            print("bt generated src-trg")
            src_str = self.src_dict.string(print_samples[0]['source'])
            tgt_str = self.tgt_dict.string(print_samples[0]['target'])
            print(src_str)
            print(tgt_str)

            return {0:self.output_collater(samples), 1: self.backward_output_collater(backward_samples), 2: self.backward_output_collater(gold_standard_samples)}
        else:
            if samples[0].get('is_dummy', False):
                return samples
            #return samples
            samples, backward_samples, _ = backtranslate_samples(
                samples=samples,
                collate_fn=self.tgt_dataset.collater,
                generate_fn=(
                    lambda net_input: self.backtranslation_fn(net_input)
                ),
                cuda=self.cuda,
                noising=self.noising,
            )
            print("bt generated src-trg")
            src_str = self.tgt_dict.string(samples[0]['source'])
            tgt_str = self.src_dict.string(samples[0]['target'])
            print(src_str)
            print(tgt_str)
            print("update bt src-trg")
            src_str = self.src_dict.string(backward_samples[0]['source'])
            tgt_str = self.tgt_dict.string(backward_samples[0]['target'])
            print(src_str)
            print(tgt_str)
            return {0:self.output_collater(samples), 1: self.backward_output_collater(backward_samples)}

    def num_tokens(self, index):
        """Just use the tgt dataset num_tokens"""
        return self.tgt_dataset.num_tokens(index)

    def ordered_indices(self):
        """Just use the tgt dataset ordered_indices"""
        return self.tgt_dataset.ordered_indices()

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used
        when filtering a dataset with ``--max-positions``.

        Note: we use *tgt_dataset* to approximate the length of the source
        sentence, since we do not know the actual length until after
        backtranslation.
        """
        if self.bt_langpair:
            return self.tgt_dataset.size(index)
        tgt_size = self.tgt_dataset.size(index)[0]
        return (tgt_size, tgt_size)

    @property
    def supports_prefetch(self):
        return getattr(self.tgt_dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.tgt_dataset.prefetch(indices)
