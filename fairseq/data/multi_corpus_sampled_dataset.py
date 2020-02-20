# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np

from . import FairseqDataset


def uniform_sampler(x, p=None):
    # Sample from uniform distribution
    return np.random.choice(x, 1, p=p).item()

class MultiCorpusSampledDataset(FairseqDataset):
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
        sampling_func: Callable[[List], int] = None,
        sample_instance = False,
        split=None,
	datasize_t=None,
        alpha_p=0,
    ):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        self.datasets = datasets
        if sampling_func is None:
            sampling_func = uniform_sampler
        self.sampling_func = sampling_func
        self.p = None

        self.sample_instance = sample_instance
        self.split = split

        self.total_num_instances = 0
        for _, dataset in datasets.items():
            assert isinstance(dataset, FairseqDataset)
            self.total_num_instances += dataset.__len__()

        self._ordered_indices = None
        if datasize_t is not None:
            self.p = np.array([len(data)**(1/datasize_t) for data in datasets.values()])
            self.p = self.p / np.sum(self.p)
            self.datasize_p = self.p
            print("data sampling with temperature {} is {}".format(datasize_t, str(self.p)) )
        else:
            self.p = np.array([1 for _ in range(len(datasets))])
            self.p = self.p / np.sum(self.p)
        self.alpha_p = alpha_p

    def __len__(self):
        """
        Length of this dataset is the sum of individual datasets
        """
        return self.total_num_instances

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

    def _map_index_to_dataset(self, key: int, index: int):
        """
        Different underlying datasets have different lengths. In order to ensure
        we are not accessing an index outside the range of the current dataset
        size, we wrap around. This function should be called after we have
        created an ordering for this and all underlying datasets.
        """
        assert (
            self._ordered_indices is not None
        ), "Must call MultiCorpusSampledDataset.ordered_indices() first"
        mapped_index = index % len(self.datasets[key])
        return self._ordered_indices[key][mapped_index]

    def __getitem__(self, index: int):
        """
        Get the item associated with index from each underlying dataset.
        Since index is in the range of [0, TotalNumInstances], we need to
        map the index to the dataset before retrieving the item.
        """
        return OrderedDict(
            [
                (key, dataset[self._map_index_to_dataset(key, index)])
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
        #print(logits)
        print("previous probs")
        print(self.p)
        for i, l in enumerate(logits):
            if logits[i] < 0:
                logits[i] = 0
        if sum(logits) == 0:
            logits = [0.1 for _ in range(len(logits))]
        p = np.array(logits) / sum(logits)
        if self.alpha_p > 0:
            #self.p = self.alpha_p * self.datasize_p + (1-self.alpha_p) * p
            self.p = np.array([i*j for i, j in zip(self.datasize_p, p) ])
            self.p = self.p / np.sum(self.p)
        else:
            self.p = p
        print("final probs")
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
        if self.sample_instance:
            collated_samples = OrderedDict([(key, []) for key in self.datasets.keys()])
            for sample in samples:
                selected_key = self.sampling_func(list(self.datasets.keys()), self.p)
                collated_samples[selected_key].append(sample[selected_key])
            for key in collated_samples.keys():
                if len(collated_samples[key]) > 0:
                    collated_samples[key] = self.datasets[key].collater(collated_samples[key])
            return collated_samples
        else:
            selected_key = self.sampling_func(list(self.datasets.keys()), self.p)
            selected_samples = [sample[selected_key] for sample in samples]
            return OrderedDict([
                (selected_key, self.datasets[selected_key].collater(selected_samples))
                ])

    def num_tokens(self, index: int):
        """
        Return an example's length (number of tokens), used for batching. Here
        we return the max across all examples at index across all underlying
        datasets.
        """
        #return max(
        #    dataset.num_tokens(self._map_index_to_dataset(key, index))
        #    for key, dataset in self.datasets.items()
        #)
        if len(self.datasets.keys()) > 1:
            k = list(self.datasets.keys())[1]
        else:
            k = list(self.datasets.keys())[0]
        return self.datasets[k].num_tokens(self._map_index_to_dataset(k, index))

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
            key: dataset.size(self._map_index_to_dataset(key, index))
            for key, dataset in self.datasets.items()
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
                [self._map_index_to_dataset(key, index) for index in indices]
            )
