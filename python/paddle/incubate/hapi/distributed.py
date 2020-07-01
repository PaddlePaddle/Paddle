# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.io import BatchSampler

__all__ = ['DistributedBatchSampler']


class DistributedBatchSampler(BatchSampler):
    """Sampler that restricts data loading to a subset of the dataset.
    In such case, each process can pass a DistributedBatchSampler instance 
    as a DataLoader sampler, and load a subset of the original dataset that 
    is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
        
    Args:
        dataset(paddle.io.Dataset): this could be a `paddle.io.Dataset` implement
                     or other python object which implemented
                     `__len__` for BatchSampler to get sample
                     number of data source.
        batch_size(int): sample indice number in a mini-batch indices.
        shuffle(bool): whther to shuffle indices order before genrating
            batch indices. Default False.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size. Default False
    Examples:
        .. code-block:: python
            
            from paddle.incubate.hapi.distributed import DistributedBatchSampler
            class FakeDataset():
                def __init__(self):
                    pass

                def __getitem__(self, idx):
                    return idx,

                def __len__(self):
                    return 10

            train_dataset = FakeDataset()
            dist_train_dataloader = DistributedBatchSampler(train_dataset, batch_size=4)
            for data in dist_train_dataloader:
                # do something
                break
    """

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        self.drop_last = drop_last
        self.nranks = ParallelEnv().nranks
        self.local_rank = ParallelEnv().local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
