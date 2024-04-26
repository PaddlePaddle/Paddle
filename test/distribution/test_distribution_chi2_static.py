# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distribution import chi2

paddle.enable_static()

np.random.seed(2024)
paddle.seed(2024)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df'),
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
                (2, 10),
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
        ),
    ],
)
class TestChi2(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            df = paddle.static.data('df', self.df.shape, self.df.dtype)
            self._paddle_chi2 = chi2.Chi2(df)
            self.feeds = {'df': self.df}

    def test_mean(self):
        with paddle.static.program_guard(self.program):
            [mean] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_chi2.mean],
            )
            np.testing.assert_allclose(
                mean,
                scipy.stats.chi2.mean(self.df),
                rtol=config.RTOL.get(str(self.df.dtype)),
                atol=config.ATOL.get(str(self.df.dtype)),
            )

    def test_variance(self):
        with paddle.static.program_guard(self.program):
            [variance] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_chi2.variance],
            )
            np.testing.assert_allclose(
                variance,
                scipy.stats.chi2.var(self.df),
                rtol=config.RTOL.get(str(self.df.dtype)),
                atol=config.ATOL.get(str(self.df.dtype)),
            )

    def test_entropy(self):
        with paddle.static.program_guard(self.program):
            [entropy] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_chi2.entropy()],
            )
            np.testing.assert_allclose(
                entropy,
                scipy.stats.chi2.entropy(self.df),
                rtol=config.RTOL.get(str(self.df.dtype)),
                atol=config.ATOL.get(str(self.df.dtype)),
            )

    def test_prob(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data(
                'value',
                self._paddle_chi2.df.shape,
                self._paddle_chi2.df.dtype,
            )
            prob = self._paddle_chi2.prob(value)

            random_number = np.random.rand(*self._paddle_chi2.df.shape).astype(
                self.df.dtype
            )
            feeds = dict(self.feeds, value=random_number)
            [prob] = self.executor.run(
                self.program, feed=feeds, fetch_list=[prob]
            )
            np.testing.assert_allclose(
                prob,
                scipy.stats.chi2.pdf(random_number, self.df),
                rtol=config.RTOL.get(str(self.df.dtype)),
                atol=config.ATOL.get(str(self.df.dtype)),
            )

    def test_log_prob(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data(
                'value',
                self._paddle_chi2.df.shape,
                self._paddle_chi2.df.dtype,
            )
            log_prob = self._paddle_chi2.log_prob(value)

            random_number = np.random.rand(*self._paddle_chi2.df.shape).astype(
                self.df.dtype
            )
            feeds = dict(self.feeds, value=random_number)
            [log_prob] = self.executor.run(
                self.program, feed=feeds, fetch_list=[log_prob]
            )
            np.testing.assert_allclose(
                log_prob,
                scipy.stats.chi2.logpdf(random_number, self.df),
                rtol=config.RTOL.get(str(self.df.dtype)),
                atol=config.ATOL.get(str(self.df.dtype)),
            )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df'),
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
class TestGammaSample(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            df = paddle.static.data('df', self.df.shape, self.df.dtype)
            self._paddle_chi2 = chi2.Chi2(df)
            self.feeds = {'df': self.df}

    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': () + np.squeeze(self.df).shape,
            },
            {
                'input': (2, 2),
                'expect': (2, 2) + np.squeeze(self.df).shape,
            },
        ]
        for case in cases:
            with paddle.static.program_guard(self.program):
                [data] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_chi2.sample(case.get('input')),
                )
                self.assertTrue(data.shape == case.get('expect'))

    def test_rsample_shape(self):
        cases = [
            {
                'input': (),
                'expect': () + np.squeeze(self.df).shape,
            },
            {
                'input': (2, 2),
                'expect': (2, 2) + np.squeeze(self.df).shape,
            },
        ]
        for case in cases:
            with paddle.static.program_guard(self.program):
                [data] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_chi2.rsample(case.get('input')),
                )

                self.assertTrue(data.shape == case.get('expect'))

    def test_sample(self):
        sample_shape = (30000,)
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_chi2.sample(sample_shape),
            )
            except_shape = sample_shape + np.squeeze(self.df).shape
            self.assertTrue(data.shape == except_shape)
            np.testing.assert_allclose(
                data.mean(axis=0),
                scipy.stats.chi2.mean(self.df),
                rtol=0.1,
                atol=config.ATOL.get(str(self.df.dtype)),
            )
            np.testing.assert_allclose(
                data.var(axis=0),
                scipy.stats.chi2.var(self.df),
                rtol=0.1,
                atol=config.ATOL.get(str(self.df.dtype)),
            )

    def test_rsample(self):
        sample_shape = (30000,)
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_chi2.rsample(sample_shape),
            )
            except_shape = sample_shape + np.squeeze(self.df).shape
            self.assertTrue(data.shape == except_shape)
            np.testing.assert_allclose(
                data.mean(axis=0),
                scipy.stats.chi2.mean(self.df),
                rtol=0.1,
                atol=config.ATOL.get(str(self.df.dtype)),
            )
            np.testing.assert_allclose(
                data.var(axis=0),
                scipy.stats.chi2.var(self.df),
                rtol=0.1,
                atol=config.ATOL.get(str(self.df.dtype)),
            )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df'),
    [
        ('0-dim', 0.4),
    ],
)
class TestChi2SampleKS(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            df = paddle.static.data('df', (), 'float')
            self._paddle_chi2 = chi2.Chi2(df)
            self.feeds = {'df': self.df}

    def test_sample(self):
        sample_shape = (15000,)
        with paddle.static.program_guard(self.program):
            [samples] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_chi2.sample(sample_shape),
            )
            self.assertTrue(self._kstest(samples))

    def test_rsample(self):
        sample_shape = (15000,)
        with paddle.static.program_guard(self.program):
            [samples] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_chi2.rsample(sample_shape),
            )
            self.assertTrue(self._kstest(samples))

    def _kstest(self, samples):
        # Uses the Kolmogorov-Smirnov test for goodness of fit.
        ks, _ = scipy.stats.kstest(samples, scipy.stats.chi2(self.df).cdf)
        return ks < 0.02


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (
        parameterize.TEST_CASE_NAME,
        'df1',
        'df2',
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
class TestChi2KL(unittest.TestCase):
    def setUp(self):
        self.program1 = paddle.static.Program()
        self.program2 = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program1, self.program2):
            df1 = paddle.static.data('df1', self.df1.shape, self.df1.dtype)
            df2 = paddle.static.data('df2', self.df2.shape, self.df2.dtype)

            self._paddle_chi2_1 = chi2.Chi2(df1)
            self._paddle_chi2_2 = chi2.Chi2(df2)
            self.feeds = {
                'df1': self.df1,
                'df2': self.df2,
            }

    def test_kl_divergence(self):
        with paddle.static.program_guard(self.program1, self.program2):
            self.executor.run(self.program2)
            [kl] = self.executor.run(
                self.program1,
                feed=self.feeds,
                fetch_list=[
                    self._paddle_chi2_1.kl_divergence(self._paddle_chi2_2)
                ],
            )
            np.testing.assert_allclose(
                kl,
                self._kl(),
                rtol=config.RTOL.get(str(self.df1.dtype)),
                atol=config.ATOL.get(str(self.df2.dtype)),
            )

    def test_kl1_error(self):
        self.assertRaises(
            TypeError,
            self._paddle_chi2_1.kl_divergence,
            paddle.distribution.beta.Beta,
        )

    def _kl(self):
        '''chi(df) = gamma(0.5*df, 0.5)'''
        concentration1 = self.df1 / 2
        concentration2 = self.df2 / 2
        rate1 = 0.5
        rate2 = 0.5
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
