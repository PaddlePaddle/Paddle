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
from distribution import config
from parameterize import (
    TEST_CASE_NAME,
    parameterize_cls,
    parameterize_func,
)

import paddle
from paddle.distribution.continuous_bernoulli import ContinuousBernoulli


class ContinuousBernoulli_np:
    def __init__(self, probs, lims=(0.48, 0.52)):
        self.lims = lims
        self.dtype = probs.dtype
        eps_prob = 1.1920928955078125e-07
        self.probs = np.clip(probs, a_min=eps_prob, a_max=1.0 - eps_prob)

    def _cut_support_region(self):
        return np.logical_or(
            np.less_equal(self.probs, self.lims[0]),
            np.greater_equal(self.probs, self.lims[1]),
        )

    def _cut_probs(self):
        return np.where(
            self._cut_support_region(),
            self.probs,
            self.lims[0] * np.ones_like(self.probs),
        )

    def _tanh_inverse(self, value):
        return 0.5 * (np.log1p(value) - np.log1p(-value))

    def _log_constant(self):
        cut_probs = self._cut_probs()
        cut_probs_below_half = np.where(
            np.less_equal(cut_probs, 0.5), cut_probs, np.zeros_like(cut_probs)
        )
        cut_probs_above_half = np.where(
            np.greater_equal(cut_probs, 0.5), cut_probs, np.ones_like(cut_probs)
        )
        log_constant_propose = np.log(
            2.0 * np.abs(self._tanh_inverse(1.0 - 2.0 * cut_probs))
        ) - np.where(
            np.less_equal(cut_probs, 0.5),
            np.log1p(-2.0 * cut_probs_below_half),
            np.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = np.square(self.probs - 0.5)
        taylor_expansion = np.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return np.where(
            self._cut_support_region(), log_constant_propose, taylor_expansion
        )

    def np_variance(self):
        cut_probs = self._cut_probs()
        tmp = np.divide(
            np.square(cut_probs) - cut_probs, np.square(1.0 - 2.0 * cut_probs)
        )
        propose = tmp + np.divide(
            1.0, np.square(2.0 * self._tanh_inverse(1.0 - 2.0 * cut_probs))
        )
        x = np.square(self.probs - 0.5)
        taylor_expansion = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        return np.where(self._cut_support_region(), propose, taylor_expansion)

    def np_mean(self):
        cut_probs = self._cut_probs()
        tmp = cut_probs / (2.0 * cut_probs - 1.0)
        propose = tmp + 1.0 / (2.0 * self._tanh_inverse(1.0 - 2.0 * cut_probs))
        x = self.probs - 0.5
        taylor_expansion = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * np.square(x)) * x
        return np.where(self._cut_support_region(), propose, taylor_expansion)

    def np_entropy(self):
        log_p = np.log(self.probs)
        log_1_minus_p = np.log1p(-self.probs)
        return np.where(
            np.equal(self.probs, 0.5),
            np.full_like(self.probs, 0.0),
            (
                -self._log_constant()
                + self.np_mean() * (log_1_minus_p - log_p)
                - log_1_minus_p
            ),
        )

    def np_prob(self, value):
        return np.exp(self.np_log_prob(value))

    def np_log_prob(self, value):
        eps = 1e-8
        cross_entropy = np.nan_to_num(
            value * np.log(self.probs) + (1.0 - value) * np.log(1 - self.probs),
            neginf=-eps,
        )
        return self._log_constant() + cross_entropy

    def np_cdf(self, value):
        cut_probs = self._cut_probs()
        cdfs = (
            np.power(cut_probs, value) * np.power(1.0 - cut_probs, 1.0 - value)
            + cut_probs
            - 1.0
        ) / (2.0 * cut_probs - 1.0)
        unbounded_cdfs = np.where(self._cut_support_region(), cdfs, value)
        return np.where(
            np.less_equal(value, 0.0),
            np.zeros_like(value),
            np.where(
                np.greater_equal(value, 1.0),
                np.ones_like(value),
                unbounded_cdfs,
            ),
        )

    def np_icdf(self, value):
        cut_probs = self._cut_probs()
        return np.where(
            self._cut_support_region(),
            (
                np.log1p(-cut_probs + value * (2.0 * cut_probs - 1.0))
                - np.log1p(-cut_probs)
            )
            / (np.log(cut_probs) - np.log1p(-cut_probs)),
            value,
        )

    def np_kl_divergence(self, other):
        part1 = -self.np_entropy()
        log_q = np.log(other.probs)
        log_1_minus_q = np.log1p(-other.probs)
        part2 = -(
            other._log_constant()
            + self.np_mean() * (log_q - log_1_minus_q)
            + log_1_minus_q
        )
        return part1 + part2


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'probs'),
    [
        ('half', np.array(0.5).astype("float32")),
        (
            'one-dim',
            parameterize.xrand((1,), min=0.0, max=1.0).astype("float64"),
        ),
        (
            'multi-dim',
            parameterize.xrand((2, 3), min=0.0, max=1.0).astype("float32"),
        ),
    ],
)
class TestContinuousBernoulli(unittest.TestCase):
    def setUp(self):
        self._dist = ContinuousBernoulli(
            probs=paddle.to_tensor(self.probs), lims=(0.48, 0.52)
        )
        self._np_dist = ContinuousBernoulli_np(self.probs, lims=(0.48, 0.52))

    def test_mean(self):
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(
            mean,
            self._np_dist.np_mean(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_variance(self):
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(
            var,
            self._np_dist.np_variance(),
            rtol=0.01,
            atol=0.0,
        )

    def test_entropy(self):
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.probs.dtype)
        np.testing.assert_allclose(
            entropy,
            self._np_dist.np_entropy(),
            rtol=0.01,
            atol=0.0,
        )

    def test_sample(self):
        sample_shape = ()
        samples = self._dist.sample(sample_shape)
        self.assertEqual(samples.numpy().dtype, self.probs.dtype)
        self.assertEqual(
            tuple(samples.shape),
            sample_shape + self._dist.batch_shape + self._dist.event_shape,
        )

        sample_shape = (50000,)
        samples = self._dist.sample(sample_shape)
        sample_mean = samples.mean(axis=0)
        sample_variance = samples.var(axis=0)

        np.testing.assert_allclose(
            sample_mean,
            self._dist.mean,
            rtol=0.1,
            atol=0.0,
        )
        np.testing.assert_allclose(
            sample_variance,
            self._dist.variance,
            rtol=0.1,
            atol=0.0,
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'probs', 'value'),
    [
        (
            'zero-dim',
            np.array(0.3).astype("float32"),
            parameterize.xrand((5,), min=0.0, max=1.0).astype("float32"),
        ),
        (
            'value-same-shape',
            parameterize.xrand((5,), min=0.0, max=1.0).astype("float32"),
            parameterize.xrand((5,), min=0.0, max=1.0).astype("float32"),
        ),
        (
            'value-broadcast-shape',
            parameterize.xrand((1,), min=0.0, max=1.0).astype("float64"),
            parameterize.xrand((2, 3), min=0.0, max=1.0).astype("float64"),
        ),
    ],
)
class TestContinuousBernoulliProbs(unittest.TestCase):
    def setUp(self):
        self._dist = ContinuousBernoulli(
            probs=paddle.to_tensor(self.probs), lims=(0.48, 0.52)
        )
        self._np_dist = ContinuousBernoulli_np(self.probs, lims=(0.48, 0.52))

    def test_prob(self):
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            self._np_dist.np_prob(self.value),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_log_prob(self):
        np.testing.assert_allclose(
            self._dist.log_prob(paddle.to_tensor(self.value)),
            self._np_dist.np_log_prob(self.value),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_cdf(self):
        np.testing.assert_allclose(
            self._dist.cdf(paddle.to_tensor(self.value)),
            self._np_dist.np_cdf(self.value),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_icdf(self):
        np.testing.assert_allclose(
            self._dist.icdf(paddle.to_tensor(self.value)),
            self._np_dist.np_icdf(self.value),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'p_1', 'p_2'),
    [
        (
            'zero-dim',
            np.array(0.2).astype("float32"),
            np.array(0.4).astype("float32"),
        ),
        (
            'one-dim',
            parameterize.xrand((1,), min=0.0, max=1.0).astype("float32"),
            parameterize.xrand((1,), min=0.0, max=1.0).astype("float32"),
        ),
        (
            'multi-dim',
            parameterize.xrand((5,), min=0.0, max=1.0).astype("float64"),
            parameterize.xrand((5,), min=0.0, max=1.0).astype("float64"),
        ),
    ],
)
class TestContinuousBernoulliKL(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self._dist1 = ContinuousBernoulli(
            probs=paddle.to_tensor(self.p_1), lims=(0.48, 0.52)
        )
        self._dist2 = ContinuousBernoulli(
            probs=paddle.to_tensor(self.p_2), lims=(0.48, 0.52)
        )
        self._np_dist1 = ContinuousBernoulli_np(self.p_1, lims=(0.48, 0.52))
        self._np_dist2 = ContinuousBernoulli_np(self.p_2, lims=(0.48, 0.52))

    def test_kl_divergence(self):
        kl0 = self._dist1.kl_divergence(self._dist2)
        kl1 = self._np_dist1.np_kl_divergence(self._np_dist2)

        self.assertEqual(tuple(kl0.shape), self._dist1.batch_shape)
        self.assertEqual(tuple(kl1.shape), self._dist1.batch_shape)
        np.testing.assert_allclose(
            kl0,
            kl1,
            rtol=0.01,
            atol=0.0,
        )


@parameterize.place(config.DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['ContinuousBernoulliTestError'])
class ContinuousBernoulliTestError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)

    @parameterize_func(
        [
            (
                paddle.to_tensor([0.3, 0.5]),
                paddle.to_tensor([0.2, 0.8, 0.6]),
            ),
        ]
    )
    def test_bad_kl_div(self, probs1, probs2):
        with paddle.base.dygraph.guard(self.place):
            rv = ContinuousBernoulli(probs1)
            rv_other = ContinuousBernoulli(probs2)
            self.assertRaises(ValueError, rv.kl_divergence, rv_other)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=3, exit=False)
