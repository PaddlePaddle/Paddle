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
from paddle.distribution.binomial import Binomial


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'total_count', 'probs'),
    [
        (
            'zero-dim',
            np.array(1000),
            np.array(0.6),
        ),
        (
            'one-dim',
            1000,
            np.array([0.4]).astype('float32'),
        ),
        (
            'multi-dim-total_count-probability',
            parameterize.xrand((2, 1), min=1, max=100).astype('int32'),
            parameterize.xrand((2, 3), dtype='float64', min=0.3, max=1),
        ),
    ],
)
class TestBinomial(unittest.TestCase):
    def setUp(self):
        self._dist = Binomial(
            total_count=paddle.to_tensor(self.total_count),
            probs=paddle.to_tensor(self.probs),
        )

    def test_mean(self):
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(
            mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_variance(self):
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(
            var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_entropy(self):
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(
            entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_sample(self):
        sample_shape = ()
        samples = self._dist.sample(sample_shape)
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

    def _np_variance(self):
        return scipy.stats.binom.var(self.total_count, self.probs)

    def _np_mean(self):
        return scipy.stats.binom.mean(self.total_count, self.probs)

    def _np_entropy(self):
        return scipy.stats.binom.entropy(self.total_count, self.probs)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'total_count', 'probs', 'value'),
    [
        (
            'value-same-shape',
            1000,
            np.array([0.12, 0.3, 0.85]).astype('float64'),
            np.array([2.0, 55.0, 999.0]).astype('float64'),
        ),
        (
            'value-broadcast-shape',
            10,
            np.array([[0.3, 0.7], [0.5, 0.5]]),
            np.array([[[4.0, 6], [8, 2]], [[2.0, 4], [9, 7]]]),
        ),
    ],
)
class TestBinomialProbs(unittest.TestCase):
    def setUp(self):
        self._dist = Binomial(
            total_count=paddle.to_tensor(self.total_count),
            probs=paddle.to_tensor(self.probs),
        )

    def test_prob(self):
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            scipy.stats.binom.pmf(self.value, self.total_count, self.probs),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_log_prob(self):
        np.testing.assert_allclose(
            self._dist.log_prob(paddle.to_tensor(self.value)),
            scipy.stats.binom.logpmf(self.value, self.total_count, self.probs),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'n_1', 'p_1', 'n_2', 'p_2'),
    [
        (
            'one-dim-probability',
            np.array([3333]),
            parameterize.xrand((1,), dtype='float32', min=0, max=1),
            np.array([3333]),
            parameterize.xrand((1,), dtype='float32', min=0, max=1),
        ),
        (
            'multi-dim-probability',
            np.array([25, 25, 25]),
            parameterize.xrand((2, 3), dtype='float64', min=0, max=1),
            np.array([25, 25, 25]),
            parameterize.xrand((2, 3), dtype='float64', min=0, max=1),
        ),
    ],
)
class TestBinomialKL(unittest.TestCase):
    def setUp(self):
        self._dist1 = Binomial(
            total_count=paddle.to_tensor(self.n_1),
            probs=paddle.to_tensor(self.p_1),
        )
        self._dist2 = Binomial(
            total_count=paddle.to_tensor(self.n_2),
            probs=paddle.to_tensor(self.p_2),
        )

    def test_kl_divergence(self):
        kl0 = self._dist1.kl_divergence(self._dist2)
        kl1 = self.kl_divergence(self._dist1, self._dist2)

        self.assertEqual(tuple(kl0.shape), self.p_1.shape)
        self.assertEqual(tuple(kl1.shape), self.p_1.shape)
        np.testing.assert_allclose(
            kl0,
            kl1,
            rtol=config.RTOL.get(str(self.p_1.dtype)),
            atol=config.ATOL.get(str(self.p_1.dtype)),
        )

    def kl_divergence(self, dist1, dist2):
        support = np.arange(1 + self.n_1.max(), dtype=self.p_1.dtype)
        support = support.reshape((-1,) + (1,) * len(self.p_1.shape))
        log_prob_1 = scipy.stats.binom.logpmf(
            support, dist1.total_count, dist1.probs
        )
        log_prob_2 = scipy.stats.binom.logpmf(
            support, dist2.total_count, dist2.probs
        )
        return (np.exp(log_prob_1) * (log_prob_1 - log_prob_2)).sum(0)


if __name__ == '__main__':
    unittest.main()
