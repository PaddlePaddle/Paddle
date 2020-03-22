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
        data_source(Dataset): this could be a `fluid.io.Dataset` 
                implement or other python object which implemented
                `__len__` for BatchSampler to get indices as the
                range of :attr:`data_source` length. Default None.
        indices (list|tuple): a substitution parameter for
                :attr:`data_source` either :attr:`data_source` or
                :attr:`indices` should be set, give the whole
                indices to sampler from directly. Default None.
        shuffle(bool): whther to shuffle indices order before genrate
                batch indices. :attr:`shuffle` should not be set if 
                :attr:`sample_iter` is set. Default False.
        batch_size(int): sample indice number in a mini-batch indices.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size. Default False

    Examples:
        
        .. code-block:: python
            
            from paddle.fluid.io import BatchSampler, MNIST

            # init with indices
            bs = BatchSampler(indices=list(range(1000)),
                              shuffle=True,
                              batch_size=8,
                              drop_last=True)

            for batch_indices in bs:
                print(batch_indices)

            # init with data_source
            bs = BatchSampler(data_source=MNIST(mode='test')),
                              shuffle=False,
                              batch_size=16,
                              drop_last=False)

            for batch_indices in bs:
                print(batch_indices)

    see `fluid.io.DataLoader`

    """

    def __init__(self,
                 data_source=None,
                 indices=None,
                 shuffle=False,
                 batch_size=1,
                 drop_last=False):
        if data_source is None:
            assert indices is not None, \
                "either data_source or indices should be set"
            assert isinstance(indices, list) or isinstance(indices, tuple), \
                "indices should be a list or tuple"
            self.indices = indices
        else:
            assert indices is None, \
                "should not set both data_source and indices"
            self.indices = list(range(len(data_source)))

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
        if self.shuffle:
            np.random.shuffle(self.indices)
        _iter = iter(self.indices)

        batch_indices = []
        for idx in _iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = len(self.indices)
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size
