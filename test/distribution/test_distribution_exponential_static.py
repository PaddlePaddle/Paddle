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
paddle.seed(2023)

paddle.enable_static()


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate'),
    [
        (
            'one-dim',
            parameterize.xrand(
                (4,),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ),
        ),
        (
            'multi-dim',
            parameterize.xrand(
                (10, 12),
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

    def test_cdf(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data(
                'value',
                self._paddle_expon.rate.shape,
                self._paddle_expon.rate.dtype,
            )
            cdf = self._paddle_expon.cdf(value)

            random_number = np.random.rand(
                *self._paddle_expon.rate.shape
            ).astype(self.rate.dtype)
            feeds = dict(self.feeds, value=random_number)
            [cdf] = self.executor.run(
                self.program, feed=feeds, fetch_list=[cdf]
            )
            np.testing.assert_allclose(
                cdf,
                scipy.stats.expon.cdf(random_number, scale=self.scale),
                rtol=config.RTOL.get(str(self.rate.dtype)),
                atol=config.ATOL.get(str(self.rate.dtype)),
            )

    def test_icdf(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data(
                'value',
                self._paddle_expon.rate.shape,
                self._paddle_expon.rate.dtype,
            )
            icdf = self._paddle_expon.icdf(value)

            random_number = np.random.rand(
                *self._paddle_expon.rate.shape
            ).astype(self.rate.dtype)
            feeds = dict(self.feeds, value=random_number)
            [icdf] = self.executor.run(
                self.program, feed=feeds, fetch_list=[icdf]
            )
            np.testing.assert_allclose(
                icdf,
                scipy.stats.expon.ppf(random_number, scale=self.scale),
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
                'expect': tuple(np.squeeze(self.rate).shape),
            },
            {
                'input': (4, 2),
                'expect': (4, 2, *np.squeeze(self.rate).shape),
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
                'expect': tuple(np.squeeze(self.rate).shape),
            },
            {
                'input': (3, 2),
                'expect': (3, 2, *np.squeeze(self.rate).shape),
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
            np.testing.assert_allclose(
                data.var(axis=0),
                scipy.stats.expon.var(scale=self.scale),
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
            np.testing.assert_allclose(
                data.var(axis=0),
                scipy.stats.expon.var(scale=self.scale),
                rtol=0.1,
                atol=config.ATOL.get(str(self.rate.dtype)),
            )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate'),
    [
        ('0-dim', 0.4),
    ],
)
class TestExponentialSampleKS(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.scale = 1 / self.rate
            rate = paddle.static.data('rate', (), 'float')
            self._paddle_expon = exponential.Exponential(rate)
            self.feeds = {'rate': self.rate}

    def test_sample(self):
        sample_shape = (10000,)
        with paddle.static.program_guard(self.program):
            [samples] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_expon.sample(sample_shape),
            )
            self.assertTrue(self._kstest(samples))

    def test_rsample(self):
        sample_shape = (10000,)
        with paddle.static.program_guard(self.program):
            [samples] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_expon.rsample(sample_shape),
            )
            self.assertTrue(self._kstest(samples))

    def _kstest(self, samples):
        # Uses the Kolmogorov-Smirnov test for goodness of fit.
        ks, _ = scipy.stats.kstest(
            samples, scipy.stats.expon(scale=self.scale).cdf
        )
        return ks < 0.02


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate1', 'rate2'),
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
class TestExponentialKL(unittest.TestCase):
    def setUp(self):
        self.program1 = paddle.static.Program()
        self.program2 = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program1, self.program2):
            rate1 = paddle.static.data(
                'rate1', self.rate1.shape, self.rate1.dtype
            )
            rate2 = paddle.static.data(
                'rate2', self.rate2.shape, self.rate2.dtype
            )

            self._expon1 = exponential.Exponential(rate1)
            self._expon2 = exponential.Exponential(rate2)

            self.feeds = {
                'rate1': self.rate1,
                'rate2': self.rate2,
            }

    def test_kl_divergence(self):
        with paddle.static.program_guard(self.program1, self.program2):
            self.executor.run(self.program2)
            [kl] = self.executor.run(
                self.program1,
                feed=self.feeds,
                fetch_list=[self._expon1.kl_divergence(self._expon2)],
            )
            np.testing.assert_allclose(
                kl,
                self._kl(),
                rtol=config.RTOL.get(str(self.rate1.dtype)),
                atol=config.ATOL.get(str(self.rate1.dtype)),
            )

    def test_kl1_error(self):
        self.assertRaises(
            TypeError,
            self._expon1.kl_divergence,
            paddle.distribution.beta.Beta,
        )

    def _kl(self):
        rate_ratio = self.rate2 / self.rate1
        t1 = -np.log(rate_ratio)
        return t1 + rate_ratio - 1


if __name__ == '__main__':
    unittest.main()
