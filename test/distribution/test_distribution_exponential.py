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
from paddle.distribution import exponential, kl

np.random.seed(2023)
paddle.seed(2023)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate'),
    [
        (
            '0-dim',
            0.5,
        ),
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
        rate = self.rate
        if not isinstance(self.rate, numbers.Real):
            rate = paddle.to_tensor(self.rate, dtype=paddle.float32)

        self.scale = 1 / rate
        self._paddle_expon = exponential.Exponential(rate)

    def test_mean(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_expon.mean,
                scipy.stats.expon.mean(scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
            )

    def test_variance(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_expon.variance,
                scipy.stats.expon.var(scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
            )

    def test_prob(self):
        value = np.random.rand(*self._paddle_expon.rate.shape)
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_expon.prob(paddle.to_tensor(value)),
                scipy.stats.expon.pdf(value, scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
            )

    def test_cdf(self):
        value = np.random.rand(*self._paddle_expon.rate.shape)
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_expon.cdf(paddle.to_tensor(value)),
                scipy.stats.expon.cdf(value, scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
            )

    def test_icdf(self):
        value = np.random.rand(*self._paddle_expon.rate.shape)
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_expon.icdf(paddle.to_tensor(value)),
                scipy.stats.expon.ppf(value, scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
            )

    def test_entropy(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_expon.entropy(),
                scipy.stats.expon.entropy(scale=self.scale),
                rtol=config.RTOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
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
        rate = self.rate
        if not isinstance(self.rate, numbers.Real):
            rate = paddle.to_tensor(self.rate, dtype=paddle.float32)

        self.scale = 1 / rate
        self._paddle_expon = exponential.Exponential(rate)

    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': tuple(paddle.squeeze(self._paddle_expon.rate).shape),
            },
            {
                'input': (3, 2),
                'expect': (
                    3,
                    2,
                    *paddle.squeeze(self._paddle_expon.rate).shape,
                ),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_expon.sample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_sample(self):
        sample_shape = (20000,)
        samples = self._paddle_expon.sample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.rate.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.expon.mean(scale=self.scale),
            rtol=0.1,
            atol=config.ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.expon.var(scale=self.scale),
            rtol=0.1,
            atol=config.ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
        )

    def test_rsample_shape(self):
        cases = [
            {
                'input': (),
                'expect': tuple(paddle.squeeze(self._paddle_expon.rate).shape),
            },
            {
                'input': (2, 5),
                'expect': (
                    2,
                    5,
                    *paddle.squeeze(self._paddle_expon.rate).shape,
                ),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_expon.rsample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_rsample(self):
        sample_shape = (20000,)
        samples = self._paddle_expon.rsample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.rate.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.expon.mean(scale=self.scale),
            rtol=0.1,
            atol=config.ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.expon.var(scale=self.scale),
            rtol=0.1,
            atol=config.ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
        )

    def test_rsample_backpropagation(self):
        sample_shape = (1000, 2)
        with paddle.base.dygraph.guard(self.place):
            self._paddle_expon.rate.stop_gradient = False
            samples = self._paddle_expon.rsample(sample_shape)
            grads = paddle.grad([samples], [self._paddle_expon.rate])
            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0].dtype, self._paddle_expon.rate.dtype)
            self.assertEqual(grads[0].shape, self._paddle_expon.rate.shape)
            axis = list(range(len(sample_shape)))
            np.testing.assert_allclose(
                -samples.sum(axis) / self._paddle_expon.rate,
                grads[0],
                rtol=config.RTOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
                atol=config.ATOL.get(
                    str(self._paddle_expon.rate.numpy().dtype)
                ),
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
        rate = paddle.to_tensor(self.rate, dtype=paddle.float32)
        self.scale = rate.reciprocal()
        self._paddle_expon = exponential.Exponential(rate)

    def test_sample_ks(self):
        sample_shape = (10000,)
        samples = self._paddle_expon.sample(sample_shape)
        self.assertTrue(self._kstest(samples))

    def test_rsample_ks(self):
        sample_shape = (10000,)
        samples = self._paddle_expon.rsample(sample_shape)
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
        self._expon1 = exponential.Exponential(paddle.to_tensor(self.rate1))
        self._expon2 = exponential.Exponential(paddle.to_tensor(self.rate2))

    def test_kl_divergence(self):
        np.testing.assert_allclose(
            kl.kl_divergence(self._expon1, self._expon2),
            self._kl(),
            rtol=config.RTOL.get(str(self._expon1.rate.numpy().dtype)),
            atol=config.ATOL.get(str(self._expon1.rate.numpy().dtype)),
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
