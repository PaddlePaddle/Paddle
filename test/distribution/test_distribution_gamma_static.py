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

paddle.enable_static()


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'concentration', 'rate'),
    [
        (
            'one-dim',
            parameterize.xrand(
                (6,),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ),
            parameterize.xrand(
                (6,),
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
            parameterize.xrand(
                (10, 12),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ),
        ),
        (
            'broadcast',
            parameterize.xrand(
                (4, 1),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ),
            parameterize.xrand(
                (4, 6),
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
class TestGammaSample(unittest.TestCase):
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
                    fetch_list=self._paddle_gamma.sample(case.get('input')),
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
                    fetch_list=self._paddle_gamma.rsample(case.get('input')),
                )

                self.assertTrue(data.shape == case.get('expect'))

    def test_sample(self):
        sample_shape = (30000,)
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_gamma.sample(sample_shape),
            )
            except_shape = sample_shape + np.squeeze(self.rate).shape
            self.assertTrue(data.shape == except_shape)
            np.testing.assert_allclose(
                data.mean(axis=0),
                scipy.stats.gamma.mean(self.concentration, scale=self.scale),
                rtol=0.1,
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )
            np.testing.assert_allclose(
                data.var(axis=0),
                scipy.stats.gamma.var(self.concentration, scale=self.scale),
                rtol=0.1,
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )

    def test_rsample(self):
        sample_shape = (30000,)
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_gamma.rsample(sample_shape),
            )
            except_shape = sample_shape + np.squeeze(self.rate).shape
            self.assertTrue(data.shape == except_shape)
            np.testing.assert_allclose(
                data.mean(axis=0),
                scipy.stats.gamma.mean(self.concentration, scale=self.scale),
                rtol=0.1,
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )
            np.testing.assert_allclose(
                data.var(axis=0),
                scipy.stats.gamma.var(self.concentration, scale=self.scale),
                rtol=0.1,
                atol=config.ATOL.get(str(self.concentration.dtype)),
            )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'concentration', 'rate'),
    [
        ('0-dim', 0.4, 0.5),
    ],
)
class TestGammaSampleKS(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.scale = 1 / self.rate
            concentration = paddle.static.data(
                'concentration',
                (),
                'float',
            )
            rate = paddle.static.data('rate', (), 'float')
            self._paddle_gamma = gamma.Gamma(concentration, rate)
            self.feeds = {
                'concentration': self.concentration,
                'rate': self.rate,
            }

    def test_sample(self):
        sample_shape = (15000,)
        with paddle.static.program_guard(self.program):
            [samples] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_gamma.sample(sample_shape),
            )
            self.assertTrue(self._kstest(samples))

    def test_rsample(self):
        sample_shape = (15000,)
        with paddle.static.program_guard(self.program):
            [samples] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_gamma.rsample(sample_shape),
            )
            self.assertTrue(self._kstest(samples))

    def _kstest(self, samples):
        # Uses the Kolmogorov-Smirnov test for goodness of fit.
        ks, _ = scipy.stats.kstest(
            samples, scipy.stats.gamma(self.concentration, scale=self.scale).cdf
        )
        return ks < 0.02


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (
        parameterize.TEST_CASE_NAME,
        'concentration1',
        'rate1',
        'concentration2',
        'rate2',
    ),
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
class TestGammaKL(unittest.TestCase):
    def setUp(self):
        self.program1 = paddle.static.Program()
        self.program2 = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program1, self.program2):
            concentration1 = paddle.static.data(
                'concentration1',
                self.concentration1.shape,
                self.concentration1.dtype,
            )
            concentration2 = paddle.static.data(
                'concentration2',
                self.concentration2.shape,
                self.concentration2.dtype,
            )
            rate1 = paddle.static.data(
                'rate1', self.rate1.shape, self.rate1.dtype
            )
            rate2 = paddle.static.data(
                'rate2', self.rate2.shape, self.rate2.dtype
            )

            self._gamma1 = gamma.Gamma(concentration1, rate1)
            self._gamma2 = gamma.Gamma(concentration2, rate2)

            self.feeds = {
                'concentration1': self.concentration1,
                'concentration2': self.concentration2,
                'rate1': self.rate1,
                'rate2': self.rate2,
            }

    def test_kl_divergence(self):
        with paddle.static.program_guard(self.program1, self.program2):
            self.executor.run(self.program2)
            [kl] = self.executor.run(
                self.program1,
                feed=self.feeds,
                fetch_list=[self._gamma1.kl_divergence(self._gamma2)],
            )
            np.testing.assert_allclose(
                kl,
                self._kl(),
                rtol=config.RTOL.get(str(self.concentration1.dtype)),
                atol=config.ATOL.get(str(self.concentration1.dtype)),
            )

    def test_kl1_error(self):
        self.assertRaises(
            TypeError,
            self._gamma1.kl_divergence,
            paddle.distribution.beta.Beta,
        )

    def _kl(self):
        concentration1 = self.concentration1
        concentration2 = self.concentration2
        rate1 = self.rate1
        rate2 = self.rate2
        t1 = concentration2 * np.log(rate1 / rate2)
        t2 = scipy.special.gammaln(concentration2) - scipy.special.gammaln(
            concentration1
        )
        t3 = (concentration1 - concentration2) * scipy.special.digamma(
            concentration1
        )
        t4 = (rate2 - rate1) * (concentration1 / rate1)
        return t1 + t2 + t3 + t4


if __name__ == '__main__':
    unittest.main()
