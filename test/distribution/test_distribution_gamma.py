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

import numbers
import unittest

import numpy as np
import parameterize
import scipy.stats
from distribution import config

import paddle
from paddle.distribution import gamma, kl

np.random.seed(2023)
paddle.seed(2023)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'concentration', 'rate'),
    [
        (
            '0-dim',
            0.5,
            0.5,
        ),
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
        concentration = self.concentration
        if not isinstance(self.concentration, numbers.Real):
            concentration = paddle.to_tensor(self.concentration)

        rate = self.rate
        if not isinstance(self.rate, numbers.Real):
            rate = paddle.to_tensor(self.rate)

        self.scale = 1 / rate
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
        value = np.random.rand(*self._paddle_gamma.rate.shape)
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_gamma.prob(paddle.to_tensor(value)),
                scipy.stats.gamma.pdf(
                    value, self.concentration, scale=self.scale
                ),
                rtol=config.RTOL.get(
                    str(self._paddle_gamma.concentration.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_gamma.concentration.numpy().dtype)
                ),
            )

    def test_log_prob(self):
        value = np.random.rand(*self._paddle_gamma.rate.shape)
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_gamma.log_prob(paddle.to_tensor(value)),
                scipy.stats.gamma.logpdf(
                    value, self.concentration, scale=self.scale
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
        concentration = self.concentration
        if not isinstance(self.concentration, numbers.Real):
            concentration = paddle.to_tensor(self.concentration)

        rate = self.rate
        if not isinstance(self.rate, numbers.Real):
            rate = paddle.to_tensor(self.rate)

        self.scale = 1 / rate
        self._paddle_gamma = gamma.Gamma(concentration, rate)

    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': tuple(paddle.squeeze(self._paddle_gamma.rate).shape),
            },
            {
                'input': (3, 2),
                'expect': (
                    3,
                    2,
                    *paddle.squeeze(self._paddle_gamma.rate).shape,
                ),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_gamma.sample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_rsample_shape(self):
        cases = [
            {
                'input': (),
                'expect': tuple(paddle.squeeze(self._paddle_gamma.rate).shape),
            },
            {
                'input': (3, 2),
                'expect': (
                    3,
                    2,
                    *paddle.squeeze(self._paddle_gamma.rate).shape,
                ),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_gamma.rsample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_sample(self):
        sample_shape = (30000,)
        samples = self._paddle_gamma.sample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.rate.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.gamma.mean(self.concentration, scale=self.scale),
            rtol=0.1,
            atol=config.ATOL.get(
                str(self._paddle_gamma.concentration.numpy().dtype)
            ),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.gamma.var(self.concentration, scale=self.scale),
            rtol=0.1,
            atol=config.ATOL.get(
                str(self._paddle_gamma.concentration.numpy().dtype)
            ),
        )

    def test_rsample(self):
        sample_shape = (30000,)
        samples = self._paddle_gamma.rsample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.rate.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.gamma.mean(self.concentration, scale=self.scale),
            rtol=0.1,
            atol=config.ATOL.get(
                str(self._paddle_gamma.concentration.numpy().dtype)
            ),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.gamma.var(self.concentration, scale=self.scale),
            rtol=0.1,
            atol=config.ATOL.get(
                str(self._paddle_gamma.concentration.numpy().dtype)
            ),
        )

    @unittest.skip("TODO: implement standard_gamma grad op.")
    def test_rsample_backpropagation(self):
        sample_shape = (1000,)
        with paddle.base.dygraph.guard(self.place):
            self._paddle_gamma.concentration.stop_gradient = False
            self._paddle_gamma.rate.stop_gradient = False
            samples = self._paddle_gamma.rsample(sample_shape)
            grads = paddle.grad(
                [samples],
                [self._paddle_gamma.concentration, self._paddle_gamma.rate],
            )
            self.assertEqual(len(grads), 2)
            self.assertEqual(
                grads[0].dtype, self._paddle_gamma.concentration.dtype
            )
            self.assertEqual(
                grads[0].shape, self._paddle_gamma.concentration.shape
            )
            self.assertEqual(grads[1].dtype, self._paddle_gamma.rate.dtype)
            self.assertEqual(grads[1].shape, self._paddle_gamma.rate.shape)

            samples.backward()
            self.assertEqual(
                list(self._paddle_gamma.concentration.gradient().shape),
                self._paddle_gamma.concentration.shape,
            )
            self.assertEqual(
                list(self._paddle_gamma.rate.gradient().shape),
                self._paddle_gamma.rate.shape,
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
        concentration = self.concentration
        if not isinstance(self.concentration, numbers.Real):
            concentration = paddle.to_tensor(self.concentration)

        rate = self.rate
        if not isinstance(self.rate, numbers.Real):
            rate = paddle.to_tensor(self.rate)

        self.scale = 1 / rate
        self._paddle_gamma = gamma.Gamma(concentration, rate)

    def test_sample_ks(self):
        sample_shape = (15000,)
        samples = self._paddle_gamma.sample(sample_shape)
        self.assertTrue(self._kstest(samples))

    def test_rsample_ks(self):
        sample_shape = (15000,)
        samples = self._paddle_gamma.rsample(sample_shape)
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
        self._gamma1 = gamma.Gamma(
            paddle.to_tensor(self.concentration1), paddle.to_tensor(self.rate1)
        )
        self._gamma2 = gamma.Gamma(
            paddle.to_tensor(self.concentration2), paddle.to_tensor(self.rate2)
        )

    def test_kl_divergence(self):
        np.testing.assert_allclose(
            kl.kl_divergence(self._gamma1, self._gamma2),
            self._kl(),
            rtol=config.RTOL.get(str(self._gamma1.concentration.numpy().dtype)),
            atol=config.ATOL.get(str(self._gamma1.concentration.numpy().dtype)),
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
