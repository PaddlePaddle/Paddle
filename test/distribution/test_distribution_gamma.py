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

import unittest

import numpy as np
import parameterize
import scipy.stats
from distribution import config

import paddle
from paddle.distribution import gamma

np.random.seed(2023)
paddle.seed(2023)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'concentration', 'rate'),
    [
        (
            'one-dim',
            parameterize.xrand(
                (2,),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ),
            parameterize.xrand(
                (2,),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ),
        ),
        (
            'multi-dim',
            parameterize.xrand(
                (2, 3),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ),
            parameterize.xrand(
                (2, 3),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ),
        ),
    ],
)
class TestExponential(unittest.TestCase):
    def setUp(self):
        concentration = paddle.to_tensor(self.concentration)
        rate = paddle.to_tensor(self.rate)
        self.scale = rate.reciprocal()

        self._paddle_gamma = gamma.Gamma(concentration, rate)

    def test_mean(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_gamma.mean,
                scipy.stats.gamma.mean(self.concentration, scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_gamma.concentration.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_gamma.concentration.numpy().dtype)
                ),
            )

    def test_variance(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_gamma.variance,
                scipy.stats.gamma.var(self.concentration, scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_gamma.concentration.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_gamma.concentration.numpy().dtype)
                ),
            )

    def test_prob(self):
        value = [np.random.rand(*self._paddle_gamma.rate.shape)]

        for v in value:
            with paddle.base.dygraph.guard(self.place):
                np.testing.assert_allclose(
                    self._paddle_gamma.prob(paddle.to_tensor(v)),
                    scipy.stats.gamma.pdf(
                        v, self.concentration, scale=self.scale
                    ),
                    rtol=config.RTOL.get(
                        str(self._paddle_gamma.concentration.numpy().dtype)
                    ),
                    atol=config.ATOL.get(
                        str(self._paddle_gamma.concentration.numpy().dtype)
                    ),
                )

    def test_log_prob(self):
        value = [np.random.rand(*self._paddle_gamma.rate.shape)]

        for v in value:
            with paddle.base.dygraph.guard(self.place):
                np.testing.assert_allclose(
                    self._paddle_gamma.log_prob(paddle.to_tensor(v)),
                    scipy.stats.gamma.logpdf(
                        v, self.concentration, scale=self.scale
                    ),
                    rtol=config.RTOL.get(
                        str(self._paddle_gamma.concentration.numpy().dtype)
                    ),
                    atol=config.ATOL.get(
                        str(self._paddle_gamma.concentration.numpy().dtype)
                    ),
                )

    def test_entropy(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_gamma.entropy(),
                scipy.stats.gamma.entropy(self.concentration, scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_gamma.concentration.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_gamma.concentration.numpy().dtype)
                ),
            )


if __name__ == '__main__':
    unittest.main()
