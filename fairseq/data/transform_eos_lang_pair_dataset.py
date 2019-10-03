# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from . import FairseqDataset
from typing import Optional
import numpy as np

class TransformEosLangPairDataset(FairseqDataset):
    """A :class:`~fairseq.data.FairseqDataset` wrapper that transform bos on
    collated samples of language pair dataset.

    Note that the transformation is applied in :func:`collater`.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset that collates sample into
            LanguagePairDataset schema
        src_eos (int): original source end-of-sentence symbol index to be replaced
        new_src_eos (int, optional): new end-of-sentence symbol index to replace source eos symbol
        tgt_bos (int, optional): original target beginning-of-sentence symbol index to be replaced
        new_tgt_bos (int, optional): new beginning-of-sentence symbol index to replace at the
            beginning of 'prev_output_tokens'
    """

    def __init__(
        self,
        dataset: FairseqDataset,
        src_eos: int,
        new_src_eos: Optional[int] = None,
        tgt_bos: Optional[int] = None,
        new_tgt_bos: Optional[int] = None,
        new_src_eos_list: Optional[list] = None,
        new_src_eos_list_probs: Optional[list] = None,
        split: Optional[str] = 'train',
    ):
        self.dataset = dataset
        self.src_eos = src_eos
        self.new_src_eos = new_src_eos
        self.tgt_bos = tgt_bos
        self.new_tgt_bos = new_tgt_bos

        self.new_src_eos_list = new_src_eos_list
        self.new_src_eos_list_probs = new_src_eos_list_probs
        self.split = split

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        samples = self.dataset.collater(samples)

        # TODO: support different padding direction
        if self.new_src_eos is not None:
            assert(samples['net_input']['src_tokens'][:, -1] != self.src_eos).sum() == 0
            if self.new_src_eos_list is None:
                samples['net_input']['src_tokens'][:, -1] = self.new_src_eos
            else:
                if self.split == 'train':
                    new_src_eos_idx = np.random.choice(len(self.new_src_eos_list), samples['net_input']['src_tokens'].size(0), p=self.new_src_eos_list_probs)
                    #samples['net_input']['src_tokens'][:, -1] = self.new_src_eos_list[new_src_eos_idx]
                    for i in range(len(new_src_eos_idx)):
                        samples['net_input']['src_tokens'][i, -1] = self.new_src_eos_list[new_src_eos_idx[i]]
                else:
                    samples['net_input']['src_tokens'][:, -1] = self.new_src_eos

        if self.new_tgt_bos is not None:
            assert (samples['net_input']['prev_output_tokens'][:, 0] != self.tgt_bos).sum() == 0
            samples['net_input']['prev_output_tokens'][:, 0] = self.new_tgt_bos
        return samples

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
