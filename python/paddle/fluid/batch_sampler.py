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

__all__ = ["BatchSampler"]


class BatchSampler(object):
    """
    A base implement of batch sampler used by `fluid.io.DataLoader`
    which yield mini-batch indices(a list/tuple with length as
    mini-batch size and holds sample indices) iterably.

    Batch sampler used by `fluid.io.DataLoader` should be a subclass
    of `fluid.io.BatchSampler`, BatchSampler subclasses should
    implement following methods:

    :math:`__iter__`: return mini-batch indices iterably.
    :math:`__len__`: get mini-batch number in an epoch.


    Args:
        data_source: this could be a `fluid.io.Dataset` implement
                     or other python object which implemented
                     `__len__` for BatchSampler to get sample
                     number of data source.
    """

    def __init__(self, data_source, batch_size, shuffle=False, drop_last=False):
        self.data_source = data_source
        self.sample_iter = None

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"
        self.drop_last = drop_last

    def __iter__(self):
        _sample_iter = self.sample_iter
        if _sample_iter is None:
            num_samples = len(self.data_source)
            indices = np.arange(num_samples).tolist()
            if self.shuffle:
                np.random.shuffle(indices)
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
        num_samples = len(self.data_source)
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size
