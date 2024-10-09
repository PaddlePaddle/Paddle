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
from paddle.distribution import chi2

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
                (2, 2),
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
                (2, 2),
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
                'expect': tuple(paddle.squeeze(self._paddle_chi2.df).shape),
            },
            {
                'input': (2, 2),
                'expect': (2, 2, *paddle.squeeze(self._paddle_chi2.df).shape),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_chi2.sample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_sample(self):
        sample_shape = (30000,)
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
        sample_shape = (15000,)
        samples = self._paddle_chi2.sample(sample_shape)
        self.assertTrue(self._kstest(samples))

    def _kstest(self, samples):
        # Uses the Kolmogorov-Smirnov test for goodness of fit.
        ks, _ = scipy.stats.kstest(samples, scipy.stats.chi2(self.df).cdf)
        return ks < 0.02


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME),
    [
        ('chi2_test_err'),
    ],
)
class Chi2TestError(unittest.TestCase):
    @parameterize.parameterize_func(
        [
            (-1.0, ValueError),  # df < 0
            ((1.0, -1.0), ValueError),  # df < 0
        ]
    )
    def test_bad_parameter(self, df, error):
        with paddle.base.dygraph.guard(self.place):
            self.assertRaises(error, chi2.Chi2, df)

    @parameterize.parameterize_func([(10,)])  # not sequence object sample shape
    def test_bad_sample_shape(self, shape):
        with paddle.base.dygraph.guard(self.place):
            _chi2 = chi2.Chi2(1.0)
            self.assertRaises(TypeError, _chi2.sample, shape)


if __name__ == '__main__':
    unittest.main()
