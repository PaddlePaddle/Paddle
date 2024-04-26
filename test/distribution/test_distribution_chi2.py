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

import numbers
import unittest

import numpy as np
import parameterize
import scipy.stats
from distribution import config

import paddle
from paddle.distribution import chi2, kl

# paddle.enable_static()

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
        df = self.df
        if not isinstance(self.df, numbers.Real):
            df = paddle.to_tensor(self.df)

        self._paddle_chi2 = chi2.Chi2(df)

    def test_mean(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_chi2.mean,
                scipy.stats.chi2.mean(self.df),
                rtol=config.RTOL.get(str(self._paddle_chi2.df.numpy().dtype)),
                atol=config.ATOL.get(str(self._paddle_chi2.df.numpy().dtype)),
            )

    def test_variance(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_chi2.variance,
                scipy.stats.chi2.var(self.df),
                rtol=config.RTOL.get(str(self._paddle_chi2.df.numpy().dtype)),
                atol=config.ATOL.get(str(self._paddle_chi2.df.numpy().dtype)),
            )

    def test_entropy(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_chi2.entropy(),
                scipy.stats.chi2.entropy(self.df),
                rtol=config.RTOL.get(str(self.df.dtype)),
                atol=config.ATOL.get(str(self.df.dtype)),
            )

    def test_prob(self):
        value = np.random.rand(*self._paddle_chi2.df.shape)
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_chi2.prob(paddle.to_tensor(value)),
                scipy.stats.chi2.pdf(value, self.df),
                rtol=config.RTOL.get(str(self.df.dtype)),
                atol=config.ATOL.get(str(self.df.dtype)),
            )

    def test_log_prob(self):
        value = np.random.rand(*self._paddle_chi2.df.shape)
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_chi2.log_prob(paddle.to_tensor(value)),
                scipy.stats.chi2.logpdf(value, self.df),
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
class TestChi2Sample(unittest.TestCase):
    def setUp(self):
        df = self.df
        if not isinstance(self.df, numbers.Real):
            df = paddle.to_tensor(self.df)

        self._paddle_chi2 = chi2.Chi2(df)

    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': ()
                + tuple(paddle.squeeze(self._paddle_chi2.df).shape),
            },
            {
                'input': (2, 2),
                'expect': (2, 2)
                + tuple(paddle.squeeze(self._paddle_chi2.df).shape),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_chi2.sample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_rsample_shape(self):
        print("self._paddle_chi2.df:", self._paddle_chi2.df)
        print(
            "paddle.squeeze(self._paddle_chi2.df):",
            paddle.squeeze(self._paddle_chi2.df),
        )
        cases = [
            {
                'input': (),
                'expect': ()
                + tuple(paddle.squeeze(self._paddle_chi2.df).shape),
            },
            {
                'input': (2, 2),
                'expect': (2, 2)
                + tuple(paddle.squeeze(self._paddle_chi2.df).shape),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_chi2.rsample(case.get('input')).shape)
                == case.get('expect')
            )
            print("case.get('expect'):", case.get('expect'))

    def test_sample(self):
        sample_shape = (300,)
        samples = self._paddle_chi2.sample(sample_shape)
        sample_values = samples.numpy()

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.chi2.mean(self.df),
            rtol=0.1,
            atol=config.ATOL.get(str(self._paddle_chi2.df.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.chi2.var(self.df),
            rtol=0.1,
            atol=config.ATOL.get(str(self._paddle_chi2.df.numpy().dtype)),
        )

    def test_rsample(self):
        sample_shape = (300,)
        samples = self._paddle_chi2.rsample(sample_shape)
        sample_values = samples.numpy()

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.chi2.mean(self.df),
            rtol=0.1,
            atol=config.ATOL.get(str(self._paddle_chi2.df.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.chi2.var(self.df),
            rtol=0.1,
            atol=config.ATOL.get(str(self._paddle_chi2.df.numpy().dtype)),
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
        df = self.df
        if not isinstance(self.df, numbers.Real):
            df = paddle.to_tensor(self.df)

        self._paddle_chi2 = chi2.Chi2(df)

    def test_sample_ks(self):
        sample_shape = (150,)
        samples = self._paddle_chi2.sample(sample_shape)
        self.assertTrue(self._kstest(samples))

    def test_rsample_ks(self):
        sample_shape = (150,)
        samples = self._paddle_chi2.rsample(sample_shape)
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
        df1 = self.df1
        df2 = self.df2
        if not isinstance(self.df1, numbers.Real):
            df1 = paddle.to_tensor(self.df1)

        if not isinstance(self.df2, numbers.Real):
            df2 = paddle.to_tensor(self.df2)

        self._paddle_chi2_1 = chi2.Chi2(df1)
        self._paddle_chi2_2 = chi2.Chi2(df2)

    def test_kl_divergence(self):
        np.testing.assert_allclose(
            kl.kl_divergence(self._paddle_chi2_1, self._paddle_chi2_2),
            self._kl(),
            rtol=config.RTOL.get(
                str(self._paddle_chi2_1.concentration.numpy().dtype)
            ),
            atol=config.ATOL.get(
                str(self._paddle_chi2_1.concentration.numpy().dtype)
            ),
        )

    def test_kl1_error(self):
        self.assertRaises(
            TypeError,
            self._paddle_chi2_1.kl_divergence,
            paddle.distribution.beta.Beta,
        )

    def _kl(self):
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
