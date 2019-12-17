# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import random

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, char_dim=None, reverse=False, src_tag_idx=-1, tgt_tag_idx=-1,
    src_tau=-1, tgt_tau=-1, src_dict=None, tgt_dict=None, 
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])

    if src_tag_idx >= 0:
        data_utils.add_tag(samples, 'source', src_tag_idx)

    if tgt_tag_idx >= 0:
        data_utils.add_tag(samples, 'target', tgt_tag_idx)

    #if random.random() < 0.5:
    #    s_key = 'target'
    #    t_key = 'source'
    #else:
    #    s_key = 'source'
    #    t_key = 'target'
    s_key = 'source'
    t_key = 'target'
 
    if type(samples[0]['source']) == list:
        # SDE
        src_length = [len(s['source']) for s in samples]
        max_len = max(src_length)
        char_sparse = []       
        for s in samples:
            key, val = [], []
            kvs = s['source']

            for i, kv in enumerate(kvs):
                key.append(torch.LongTensor([[i for _ in range(len(kv.keys()))], list(kv.keys())]))
                val.extend(list(kv.values()))
            key = torch.cat(key, dim=1)
            val = torch.FloatTensor(val)
            sent_sparse = torch.sparse.FloatTensor(key, val, torch.Size([max_len, char_dim]))
            char_sparse.append(sent_sparse)

        src_length = np.array(src_length)
        char_sparse = np.array(char_sparse)
        sort_order = np.argsort(-src_length)

        src_tokens = char_sparse[sort_order].tolist() 
        src_lengths = torch.LongTensor(src_length[sort_order])
        sort_order = torch.LongTensor(sort_order)
    else:
        src_tokens = merge(s_key, left_pad=left_pad_source)
        # sort by descending source length
        src_lengths = torch.LongTensor([s[s_key].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get(t_key, None) is not None:
        target = merge(t_key, left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s[t_key]) for s in samples)
        target_lengths = torch.LongTensor([s[t_key].numel() for s in samples])

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                t_key,
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s[s_key]) for s in samples)
    
    # sample augmented data based on switchout
    if src_tau >= 0:
        src_tokens = data_utils.switchout(src_tokens, src_lengths, src_tau, src_dict)
    if tgt_tau >= 0:
        target = data_utils.switchout(target, target_lengths, tgt_tau, tgt_dict)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        src_tag=None, tgt_tag=None, src_tau=-1, tgt_tau=-1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_tau = src_tau
        self.tgt_tau = tgt_tau
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.src_tag = src_tag
        self.tgt_tag = tgt_tag
        if self.src_tag is not None:
            self.src_tag_idx = self.src_dict.add_symbol(self.src_tag)
            self.src_dict.add_symbol(self.tgt_tag)
        else:
            self.src_tag_idx = -1
        if self.tgt_tag is not None:
            self.tgt_dict.add_symbol(self.src_tag)
            self.tgt_tag_idx = self.tgt_dict.add_symbol(self.tgt_tag)
        else:
            self.tgt_tag_idx = -1

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, char_dim=len(self.src_dict),
            src_tag_idx=self.src_tag_idx, tgt_tag_idx=self.tgt_tag_idx,
            src_tau=self.src_tau, tgt_tau=self.tgt_tau, src_dict=self.src_dict, tgt_dict=self.tgt_dict,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
