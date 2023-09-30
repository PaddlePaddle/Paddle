# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import distribution
from paddle.distribution import exponential_family


class Exponential(exponential_family.ExponentialFamily):
    def __init__(self, rate):
        self.rate = rate
        super().__init__(self.rate.shape)

    @property
    def mean(self):
        return self.rate.reciprocal()

    @property
    def variance(self):
        return self.rate.pow(-2)

    def sample(self, shape=()):
        return self.rsample(shape)

    def rsample(self, shape=()):
        shape = distribution.Distribution._extend_shape(
            self, sample_shape=shape
        )

        uniform = paddle.uniform(
            shape=shape,
            min=float(np.finfo(dtype='float32').tiny),
            max=1.0,
            dtype=self.rate.dtype,
        )

        return -paddle.log(uniform) / self.rate

    def prob(self, value):
        return self.rate * paddle.exp(-self.rate * value)

    def log_prob(self, value):
        return paddle.log(self.rate) - self.rate * value

    def entropy(self):
        return 1.0 - paddle.log(self.rate)

    @property
    def _natural_params(self):
        return (-self.rate,)

    def _log_normalizer(self, x):
        return -paddle.log(-x)
