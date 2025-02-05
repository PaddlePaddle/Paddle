# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distribution import Poisson


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate'),
    [
        ('zero-dim', np.array(100.0).astype('float64')),
        ('one-dim', np.array([100.0]).astype('float64')),
        # boundary case and extreme case (`scipy.stats.poisson.entropy` cannot converge for very extreme cases such as rate=10000.0)
        ('multi-dim', np.array([0.0, 1000.0]).astype('float32')),
    ],
)
class TestPoisson(unittest.TestCase):
    def setUp(self):
        self._dist = Poisson(rate=paddle.to_tensor(self.rate))

    def test_mean(self):
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.rate.dtype)
        np.testing.assert_allclose(
            mean,
            scipy.stats.poisson.mean(self.rate),
            rtol=config.RTOL.get(str(self.rate.dtype)),
            atol=config.ATOL.get(str(self.rate.dtype)),
        )

    def test_variance(self):
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.rate.dtype)
        np.testing.assert_allclose(
            var,
            scipy.stats.poisson.var(self.rate),
            rtol=config.RTOL.get(str(self.rate.dtype)),
            atol=config.ATOL.get(str(self.rate.dtype)),
        )

    def test_entropy(self):
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.rate.dtype)
        np.testing.assert_allclose(
            entropy,
            scipy.stats.poisson.entropy(self.rate),
            rtol=config.RTOL.get(str(self.rate.dtype)),
            atol=config.ATOL.get(str(self.rate.dtype)),
        )

    def test_sample(self):
        sample_shape = ()
        samples = self._dist.sample(sample_shape)
        self.assertEqual(samples.numpy().dtype, self.rate.dtype)
        self.assertEqual(
            tuple(samples.shape),
            sample_shape + self._dist.batch_shape + self._dist.event_shape,
        )

        sample_shape = (5000,)
        samples = self._dist.sample(sample_shape)
        sample_mean = samples.mean(axis=0)
        sample_variance = samples.var(axis=0)

        np.testing.assert_allclose(
            sample_mean, self._dist.mean, atol=0, rtol=0.20
        )
        np.testing.assert_allclose(
            sample_variance, self._dist.variance, atol=0, rtol=0.20
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate', 'value'),
    [
        (
            'value-same-shape',
            np.array(1000).astype('float32'),
            np.array(1100).astype('float32'),
        ),
        (
            'value-broadcast-shape',
            np.array(10).astype('float64'),
            np.array([2.0, 3.0, 5.0, 10.0, 20.0]).astype('float64'),
        ),
    ],
)
class TestPoissonProbs(unittest.TestCase):
    def setUp(self):
        self._dist = Poisson(rate=paddle.to_tensor(self.rate))

    def test_prob(self):
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            scipy.stats.poisson.pmf(self.value, self.rate),
            rtol=config.RTOL.get(str(self.rate.dtype)),
            atol=config.ATOL.get(str(self.rate.dtype)),
        )

    def test_log_prob(self):
        np.testing.assert_allclose(
            self._dist.log_prob(paddle.to_tensor(self.value)),
            scipy.stats.poisson.logpmf(self.value, self.rate),
            rtol=config.RTOL.get(str(self.rate.dtype)),
            atol=config.ATOL.get(str(self.rate.dtype)),
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate_1', 'rate_2'),
    [
        (
            'zero-dim',
            parameterize.xrand((1,), min=1, max=20)
            .astype('int32')
            .astype('float64')
            .reshape([]),
            parameterize.xrand((1,), min=1, max=20)
            .astype('int32')
            .astype('float64')
            .reshape([]),
        ),
        (
            'one-dim',
            parameterize.xrand((1,), min=1, max=20)
            .astype('int32')
            .astype('float64'),
            parameterize.xrand((1,), min=1, max=20)
            .astype('int32')
            .astype('float64'),
        ),
        (
            'multi-dim',
            parameterize.xrand((5, 3), min=1, max=20)
            .astype('int32')
            .astype('float32'),
            parameterize.xrand((5, 3), min=1, max=20)
            .astype('int32')
            .astype('float32'),
        ),
    ],
)
class TestPoissonKL(unittest.TestCase):
    def setUp(self):
        self._dist1 = Poisson(rate=paddle.to_tensor(self.rate_1))
        self._dist2 = Poisson(rate=paddle.to_tensor(self.rate_2))

    def test_kl_divergence(self):
        kl0 = self._dist1.kl_divergence(self._dist2)
        kl1 = self.kl_divergence_scipy()

        self.assertEqual(tuple(kl0.shape), self._dist1.batch_shape)
        self.assertEqual(tuple(kl1.shape), self._dist1.batch_shape)
        np.testing.assert_allclose(
            kl0,
            kl1,
            rtol=config.RTOL.get(str(self.rate_1.dtype)),
            atol=config.ATOL.get(str(self.rate_1.dtype)),
        )

    def kl_divergence_scipy(self):
        rate_max = np.max(np.maximum(self.rate_1, self.rate_2))
        rate_min = np.min(np.minimum(self.rate_1, self.rate_2))
        support_max = self.enumerate_bounded_support(rate_max)
        support_min = self.enumerate_bounded_support(rate_min)
        a_min = np.min(support_min)
        a_max = np.max(support_max)
        common_support = np.arange(
            a_min, a_max, dtype=self.rate_1.dtype
        ).reshape((-1,) + (1,) * len(self.rate_1.shape))
        log_prob_1 = scipy.stats.poisson.logpmf(common_support, self.rate_1)
        log_prob_2 = scipy.stats.poisson.logpmf(common_support, self.rate_2)
        return (np.exp(log_prob_1) * (log_prob_1 - log_prob_2)).sum(0)

    def enumerate_bounded_support(self, rate):
        s = np.sqrt(rate)
        upper = int(rate + 30 * s)
        lower = int(np.clip(rate - 30 * s, a_min=0, a_max=rate))
        values = np.arange(lower, upper, dtype=self.rate_1.dtype)
        return values


if __name__ == '__main__':
    unittest.main()
