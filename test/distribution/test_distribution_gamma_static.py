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

paddle.enable_static()


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
        (
            'broadcast',
            parameterize.xrand(
                (2, 1),
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
class TestGamma(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.scale = 1 / self.rate
            concentration = paddle.static.data(
                'concentration',
                self.concentration.shape,
                self.concentration.dtype,
            )
            rate = paddle.static.data('rate', self.rate.shape, self.rate.dtype)
            self._paddle_gamma = gamma.Gamma(concentration, rate)
            self.feeds = {
                'concentration': self.concentration,
                'rate': self.rate,
            }

    def test_mean(self):
        with paddle.static.program_guard(self.program):
            [mean] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_gamma.mean],
            )
            np.testing.assert_allclose(
                mean,
                scipy.stats.gamma.mean(self.concentration, scale=self.scale),
                rtol=config.RTOL.get(str(self.concentration.dtype)),
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )

    def test_variance(self):
        with paddle.static.program_guard(self.program):
            [variance] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_gamma.variance],
            )
            np.testing.assert_allclose(
                variance,
                scipy.stats.gamma.var(self.concentration, scale=self.scale),
                rtol=config.RTOL.get(str(self.concentration.dtype)),
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )

    def test_entropy(self):
        with paddle.static.program_guard(self.program):
            [entropy] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_gamma.entropy()],
            )
            np.testing.assert_allclose(
                entropy,
                scipy.stats.gamma.entropy(self.concentration, scale=self.scale),
                rtol=config.RTOL.get(str(self.concentration.dtype)),
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )

    def test_prob(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data(
                'value',
                self._paddle_gamma.concentration.shape,
                self._paddle_gamma.concentration.dtype,
            )
            prob = self._paddle_gamma.prob(value)

            random_number = np.random.rand(
                *self._paddle_gamma.concentration.shape
            ).astype(self.concentration.dtype)
            feeds = dict(self.feeds, value=random_number)
            [prob] = self.executor.run(
                self.program, feed=feeds, fetch_list=[prob]
            )
            np.testing.assert_allclose(
                prob,
                scipy.stats.gamma.pdf(
                    random_number, self.concentration, scale=self.scale
                ),
                rtol=config.RTOL.get(str(self.concentration.dtype)),
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )

    def test_log_prob(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data(
                'value',
                self._paddle_gamma.concentration.shape,
                self._paddle_gamma.concentration.dtype,
            )
            log_prob = self._paddle_gamma.log_prob(value)

            random_number = np.random.rand(
                *self._paddle_gamma.concentration.shape
            ).astype(self.concentration.dtype)
            feeds = dict(self.feeds, value=random_number)
            [log_prob] = self.executor.run(
                self.program, feed=feeds, fetch_list=[log_prob]
            )
            np.testing.assert_allclose(
                log_prob,
                scipy.stats.gamma.logpdf(
                    random_number, self.concentration, scale=self.scale
                ),
                rtol=config.RTOL.get(str(self.concentration.dtype)),
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )


if __name__ == '__main__':
    unittest.main()
