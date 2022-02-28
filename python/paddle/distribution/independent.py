# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distribution import tool
from paddle.distribution import distribution


class Independent(distribution.Distribution):
    def __init__(self, base, reinterpreted_batch_ndims):
        if not isinstance(base, distribution.Distribution):
            raise TypeError(
                f"Expected type of 'base' is Distribution, but got {type(base)}")
        if not (0 < reinterpreted_batch_ndims <= len(base.batch_shape)):
            raise ValueError(
                f"Expected 0 < reinterpreted_batch_ndims <= {len(base.batch_shape)}, but got {reinterpreted_batch_ndims}"
            )
        self._base = base
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

        shape = base.batch_shape + base.event_shape
        super(Independent, self).__init__(
            batch_shape=shape[:len(base.batch_shape) -
                              reinterpreted_batch_ndims],
            event_shape=shape[len(base.batch_shape) -
                              reinterpreted_batch_ndims:])

    def _expand(self, batch_shape):
        raise NotImplementedError()

    @property
    def mean(self):
        return self._base.mean

    @property
    def variance(self):
        return self._base.variance

    def sample(self, shape=()):
        return self._base.sample(shape)

    def rsample(self, shape=()):
        return self._base.rsample(shape)

    def log_prob(self, value):
        return tool._sum_rightmost(
            self.base.log_prob(value), self._reinterpreted_batch_ndims)

    def prob(self, value):
        return self.log_prob(value).exp()

    def entropy(self):
        return tool._sum_rightmost(self._base.entropy(),
                                   self._reinterpreted_batch_ndims)
