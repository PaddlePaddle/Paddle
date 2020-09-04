#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import division

import numpy as np

__all__ = ["Sampler", "SequenceSampler", "RandomSampler"]


class Sampler(object):
    """
    An abstract class to encapsulate methods and behaviors of samplers.

    All sampler used by :code:`paddle.io.BatchSampler` should be a subclass
    of :code:`paddle.io.Sampler`, BatchSampler subclasses should
    implement following methods:

    :code:`__iter__`: return sample index iterably, which iterate over indices
    of dataset elements

    :code:`__len__`: the number of sample in :attr:`data_source`


    Args:
        data_source(Dataset, optional): this could be an instance of
                :code:`paddle.io.Dataset` other Python object which
                implemented :code:`__len__` for Sampler to get indices
                as the range of :attr:`dataset` length. Default None.

    Returns:
        Sampler: an iterable object for sample indices iterating

    Examples:
        
        .. code-block:: python
            
            from paddle.io import Dataset, Sampler

            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label
                
                def __len__(self):
                    return self.num_samples

            class MySampler(Sampler):
                def __init__(self, data_source):
                    self.data_source = data_source

                def __iter__(self):
                    return iter(range(len(self.data_source)))

                def __len__(self):
                    return len(self.data_source)
            
            sampler = MySampler(data_source=RandomDataset(100))

            for index in sampler:
                print(index)

    see `paddle.io.BatchSampler`
    see `paddle.io.DataLoader`

    """

    def __init__(self, data_source=None):
        self.data_source = data_source

    def __iter__(self):
        raise NotImplementedError

    # Not define __len__ method in this base class here for __len__
    # is not needed in same sence, e.g. paddle.io.IterableDataset


class SequenceSampler(Sampler):
    """
    Iterate samples sequentially, yield :code:`0, 1, 2, ..., len(data_source) -1`
    generally,

    Args:
        data_source(Dataset): dataset to sample, this could be an
                instance of :code:`paddle.io.Dataset` other Python
                object which implemented :code:`__len__`.

    Returns:
        Sampler: a Sampler yield sample index sequentially

    Examples:

        .. code-block:: python
            
            from paddle.io import Dataset, SequenceSampler

            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label
                
                def __len__(self):
                    return self.num_samples

            sampler = SequenceSampler(data_source=RandomDataset(100))

            for index in sampler:
                print(index)

    see `paddle.io.Sampler`
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """
    Iterate samples randomly, yield shuffled indices, if :attr:`replacement=False`,
    yield shuffled indices of the whole data souce, if :attr:`replacement=True`,
    :attr:`num_samples` can set to specify the sample number to draw.

    Args:
        data_source(Dataset): dataset to sample, this could be an
                instance of :code:`paddle.io.Dataset` other Python
                object which implemented :code:`__len__`.
        replacement(bool): If False, sample the whole dataset, If False,
                set :attr:`num_samples` for how many sample to draw. Default False.
        num_samples(int): set sample number to draw if :attr:`replacement`
                is True. Default None.
        generator(Generator): specify a generator to sample the data source. Default None
        
    Returns:
        Sampler: a Sampler yield sample index randomly

    Examples:

        .. code-block:: python
            
            from paddle.io import Dataset, RandomSampler

            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label
                
                def __len__(self):
                    return self.num_samples

            sampler = RandomSampler(data_source=RandomDataset(100))

            for index in sampler:
                print(index)

    see `paddle.io.Sampler`
    """

    def __init__(self,
                 data_source,
                 replacement=False,
                 num_samples=None,
                 generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("expect boolean value for replacement, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError(
                "num_samples should not be specified while replacement is False")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer, "
                             "but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator:
            for i in range(self.num_samples):
                try:
                    index = next(self.generator)
                except StopIteration:
                    return
                yield index
        else:
            if self.replacement:
                for index in np.random.choice(
                        np.arange(n), self.num_samples, replace=True).tolist():
                    yield index
            else:
                for index in np.random.choice(
                        np.arange(n), n, replace=False).tolist():
                    yield index

    def __len__(self):
        return self.num_samples
