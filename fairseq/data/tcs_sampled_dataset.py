# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np

from . import FairseqDataset


def uniform_sampler(x, sample, p=None):
    #none_mask = [True if s is None else False for s in sample.values()]
    #prob = np.where(np.array(none_mask), 0, p) 
    #prob = prob / prob.sum()
    prob, s = [], 0
    for i, d in enumerate(sample.values()):
        if d is None:
            prob.append(0)
        else:
            prob.append(p[i])
            s += p[i]
    prob = [i/s for i in prob]
    return np.random.choice(x, 1, p=prob).item()

class TCSSampledDataset(FairseqDataset):
    """
    Stores multiple instances of FairseqDataset together and in every iteration
    creates a batch by first sampling a dataset according to a specified
    probability distribution and then getting instances from that dataset.

    Args:
        datasets: an OrderedDict of FairseqDataset instances.
        sampling_func: A function for sampling over list of dataset keys.
                Default strategy is to sample uniformly.
    """

    def __init__(
        self,
        datasets: Dict[str, FairseqDataset],
        lan_dists: List,
        data_condition: str = 'target',
        sampling_func: Callable[[List], int] = None,
        sample_instance = False,
        split=None,
    ):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        self.datasets = datasets
        if sampling_func is None:
            sampling_func = uniform_sampler
        self.sampling_func = sampling_func
        #self.p = np.array([float(x) for x in lan_dists.split(",")])
        self.p = np.array(lan_dists)
        self.p = self.p / np.sum(self.p)

        assert len(self.p) == len(self.datasets)

        self.sample_instance = sample_instance
        self.split = split
        # whether it is source or target conditioned
        self.data_condition = data_condition

        self._ordered_indices = None
        self.unique_data_len = 0
        self.ordered_indices()
        self._map_indices()

    def __len__(self):
        """
        Length of this dataset is the sum of individual datasets
        """
        return self.unique_data_len

    def _map_indices(self):
        """
        Build index dict for languages based on the shared source or target
        {index: {lan_name: data_index}}
        """
        self.index_dict = {}
        # {data_conditioned: index}
        data_conditions = {}
        for lan_name, dataset in self.datasets.items():
            for i, sample in enumerate(dataset):
                # sample: {id: index, source: src_item, target: trg_item}
                if self.data_condition == "source":
                    data_key = sample['source']
                else:
                    data_key = sample['target']
                data_key = tuple(data_key.tolist())
                if data_key in data_conditions:
                    idx = data_conditions[data_key]
                    self.index_dict[idx][lan_name] = i
                else:
                    data_conditions[data_key] = len(self.index_dict)
                    self.index_dict[len(self.index_dict)] = {lan_name: i}
        self.unique_data_len = len(self.index_dict)

    def ordered_indices(self):
        """
        Ordered indices for batching. Here we call the underlying
        dataset's ordered_indices() so that we get the same random ordering
        as we would have from using the underlying dataset directly.
        """
        if self._ordered_indices is None:
            self._ordered_indices = OrderedDict(
                [
                    (key, dataset.ordered_indices())
                    for key, dataset in self.datasets.items()
                ]
            )
        return np.arange(len(self))

    def __getitem__(self, index: int):
        """
        """
        return OrderedDict(
            [
                (key, dataset[self.index_dict[index][key]]) if key in self.index_dict[index] else (key, None)
                for key, dataset in self.datasets.items()
            ]
        )
    
    def get_sample_with_key(self, key, num=8, max_count=1200):
        """
        Get some samples with a given key
        """
        dataset = self.datasets[key]
        sample_indices = np.random.choice(np.arange(len(dataset)), size=num)
        samples, count = [], 0
        for i in sample_indices:
            samples.append(dataset[i])
            count += dataset.num_tokens(i)
            if count >= max_count: break
        
        return OrderedDict([
            (key, self.datasets[key].collater(samples))
            ])

    def update_sampling_distribution(self, logits):
        print(logits)
        for i, l in enumerate(logits):
            if logits[i] < 0:
                logits[i] = 0
        if sum(logits) == 0:
            logits = [0.1 for _ in range(len(logits))]
        self.p = np.array(logits) / sum(logits)
        print("Updating probs")
        print(self.p)

    def collater(self, samples: List[Dict]):
        """
        Generate a mini-batch for this dataset.
        To convert this into a regular mini-batch we use the following
        logic:
            1. Select a dataset using the specified probability distribution.
            2. Call the collater function of the selected dataset.
        """
        if len(samples) == 0:
            return None
        collated_samples = OrderedDict([(key, []) for key in self.datasets.keys()])
        #none_mask = []
        #for sample in samples:
        #    none_mask.append(np.array(sample.values())==None)
        #none_mask = torch.LongTensor(none_mask)
        #probs = torch.LongTensor(self.p).view(1, -1).repeat(len(samples), 1)
        #probs.masked_fill_(none_mask, 0)
        #probs = probs / probs.sum(dim=-1, keepdim=True)
        #selected_keys = torch.Categorical(probs)
        for i, sample in enumerate(samples):
            selected_key = self.sampling_func(list(self.datasets.keys()), sample, self.p)
            #selected_key = self.datasets.keys()[selected_keys[i].item()]
            collated_samples[selected_key].append(sample[selected_key])
        for key in collated_samples.keys():
            if len(collated_samples[key]) > 0:
                collated_samples[key] = self.datasets[key].collater(collated_samples[key])
        return collated_samples

    def num_tokens(self, index: int):
        """
        Return an example's length (number of tokens), used for batching. Here
        we return the max across all examples at index across all underlying
        datasets.
        """
        return max(
            dataset.num_tokens(self.index_dict[index][key]) if key in self.index_dict[index] else 0
            for key, dataset in self.datasets.items()
        )

    #def size(self, index: int):
    #    """
    #    Return an example's size as a float or tuple. Here we return the max
    #    across all underlying datasets. This value is used when filtering a
    #    dataset with max-positions.
    #    """
    #    return max(
    #        dataset.size(self._map_index_to_dataset(key, index))
    #        for key, dataset in self.datasets.items()
    #    )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return {
            key: dataset.size(self.index_dict[index][key])
            for key, dataset in self.datasets.items() if key in self.index_dict[index]
        }


    @property
    def supports_prefetch(self):
        return all(
            getattr(dataset, "supports_prefetch", False)
            for dataset in self.datasets.values()
        )

    def prefetch(self, indices):
        for key, dataset in self.datasets.items():
            dataset.prefetch(
                [self.index_dict[index][key] for index in indices if key in self.index_dict[index]]
            )
