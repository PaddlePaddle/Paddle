# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class Exponential(paddle.distribution.ExponentialFamily):
    """mock exponential distribution, which support computing entropy and
       kl use bregman divergence
    """
    _mean_carrier_measure = 0

    def __init__(self, rate):
        self._rate = rate
        super(Exponential, self).__init__(batch_shape=rate.shape)

    @property
    def rate(self):
        return self._rate

    def entropy(self):
        return 1.0 - paddle.log(self._rate)

    @property
    def _natural_parameters(self):
        return (-self._rate, )

    def _log_normalizer(self, x):
        return -paddle.log(-x)


class DummyExpFamily(paddle.distribution.ExponentialFamily):
    """dummy class extend from exponential family
    """

    def __init__(self, *args):
        pass

    def entropy(self):
        return 1.0

    @property
    def _natural_parameters(self):
        return (1.0, )

    def _log_normalizer(self, x):
        return -paddle.log(-x)


@paddle.distribution.register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p, q):
    rate_ratio = q.rate / p.rate
    t1 = -rate_ratio.log()
    return t1 + rate_ratio - 1
