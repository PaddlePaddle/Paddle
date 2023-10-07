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
from paddle.distribution import exponential

np.random.seed(2023)

paddle.enable_static()


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate'),
    [
        (
            'one-dim',
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
        ),
    ],
)
class TestExponential(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.scale = 1 / self.rate
            rate = paddle.static.data('rate', self.rate.shape, self.rate.dtype)
            self._paddle_expon = exponential.Exponential(rate)
            self.feeds = {'rate': self.rate}

    def test_mean(self):
        with paddle.static.program_guard(self.program):
            [mean] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_expon.mean],
            )
            np.testing.assert_allclose(
                mean,
                scipy.stats.expon.mean(scale=self.scale),
                rtol=config.RTOL.get(str(self.rate.dtype)),
                atol=config.ATOL.get(str(self.rate.dtype)),
            )

    def test_variance(self):
        with paddle.static.program_guard(self.program):
            [variance] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_expon.variance],
            )
            np.testing.assert_allclose(
                variance,
                scipy.stats.expon.var(scale=self.scale),
                rtol=config.RTOL.get(str(self.rate.dtype)),
                atol=config.ATOL.get(str(self.rate.dtype)),
            )

    def test_entropy(self):
        with paddle.static.program_guard(self.program):
            [entropy] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_expon.entropy()],
            )
            np.testing.assert_allclose(
                entropy,
                scipy.stats.expon.entropy(scale=self.scale),
                rtol=config.RTOL.get(str(self.rate.dtype)),
                atol=config.ATOL.get(str(self.rate.dtype)),
            )

    def test_prob(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data(
                'value',
                self._paddle_expon.rate.shape,
                self._paddle_expon.rate.dtype,
            )
            prob = self._paddle_expon.prob(value)

            random_number = np.random.rand(
                *self._paddle_expon.rate.shape
            ).astype(self.rate.dtype)
            feeds = dict(self.feeds, value=random_number)
            [prob] = self.executor.run(
                self.program, feed=feeds, fetch_list=[prob]
            )
            np.testing.assert_allclose(
                prob,
                scipy.stats.expon.pdf(random_number, scale=self.scale),
                rtol=config.RTOL.get(str(self.rate.dtype)),
                atol=config.ATOL.get(str(self.rate.dtype)),
            )

    def test_log_prob(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data(
                'value',
                self._paddle_expon.rate.shape,
                self._paddle_expon.rate.dtype,
            )
            log_prob = self._paddle_expon.log_prob(value)

            random_number = np.random.rand(
                *self._paddle_expon.rate.shape
            ).astype(self.rate.dtype)
            feeds = dict(self.feeds, value=random_number)
            [log_prob] = self.executor.run(
                self.program, feed=feeds, fetch_list=[log_prob]
            )
            np.testing.assert_allclose(
                log_prob,
                scipy.stats.expon.logpdf(random_number, scale=self.scale),
                rtol=config.RTOL.get(str(self.rate.dtype)),
                atol=config.ATOL.get(str(self.rate.dtype)),
            )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate'),
    [
        (
            'one-dim',
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
        ),
    ],
)
class TestExponentialSample(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.scale = 1 / self.rate
            rate = paddle.static.data('rate', self.rate.shape, self.rate.dtype)
            self._paddle_expon = exponential.Exponential(rate)
            self.feeds = {'rate': self.rate}

    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': () + np.squeeze(self.rate).shape,
            },
            {
                'input': (4, 2),
                'expect': (4, 2) + np.squeeze(self.rate).shape,
            },
        ]
        for case in cases:
            with paddle.static.program_guard(self.program):
                [data] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_expon.sample(case.get('input')),
                )

                self.assertTrue(data.shape == case.get('expect'))

    def test_rsample_shape(self):
        cases = [
            {
                'input': (),
                'expect': () + np.squeeze(self.rate).shape,
            },
            {
                'input': (4, 2),
                'expect': (4, 2) + np.squeeze(self.rate).shape,
            },
        ]
        for case in cases:
            with paddle.static.program_guard(self.program):
                [data] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_expon.rsample(case.get('input')),
                )

                self.assertTrue(data.shape == case.get('expect'))

    def test_sample(self):
        sample_shape = (20000,)
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_expon.sample(sample_shape),
            )
            except_shape = sample_shape + np.squeeze(self.rate).shape
            self.assertTrue(data.shape == except_shape)
            np.testing.assert_allclose(
                data.mean(axis=0),
                scipy.stats.expon.mean(scale=self.scale),
                rtol=0.1,
                atol=config.ATOL.get(str(self.rate.dtype)),
            )

    def test_rsample(self):
        sample_shape = (20000,)
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_expon.rsample(sample_shape),
            )
            except_shape = sample_shape + np.squeeze(self.rate).shape
            self.assertTrue(data.shape == except_shape)
            np.testing.assert_allclose(
                data.mean(axis=0),
                scipy.stats.expon.mean(scale=self.scale),
                rtol=0.1,
                atol=config.ATOL.get(str(self.rate.dtype)),
            )


if __name__ == '__main__':
    unittest.main()
