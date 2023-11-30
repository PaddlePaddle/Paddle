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

import paddle
from paddle.distribution.continuous_bernoulli import ContinuousBernoulli


class ContinuousBernoulli_np:
    def __init__(self, probability, eps=1e-4):
        self.eps = eps
        self.dtype = 'float32'
        eps_prob = 1.1920928955078125e-07
        self.probability = np.clip(
            probability, a_min=eps_prob, a_max=1 - eps_prob
        )

    def _cut_support_region(self):
        return np.logical_or(
            np.less_equal(self.probability, 0.5 - self.eps),
            np.greater_equal(self.probability, 0.5 + self.eps),
        )

    def _cut_probs(self):
        return np.where(
            self._cut_support_region(),
            self.probability,
            (0.5 - self.eps) * np.ones_like(self.probability),
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
        x = np.square(self.probability - 0.5)
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
        x = np.square(self.probability - 0.5)
        taylor_expansion = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        return np.where(self._cut_support_region(), propose, taylor_expansion)

    def np_mean(self):
        cut_probs = self._cut_probs()
        tmp = cut_probs / (2.0 * cut_probs - 1.0)
        propose = tmp + 1.0 / (2.0 * self._tanh_inverse(1.0 - 2.0 * cut_probs))
        x = self.probability - 0.5
        taylor_expansion = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * np.square(x)) * x
        return np.where(self._cut_support_region(), propose, taylor_expansion)

    def np_entropy(self):
        log_p = np.log(self.probability)
        log_1_minus_p = np.log1p(-self.probability)
        return (
            -self._log_constant()
            + self.np_mean() * (log_1_minus_p - log_p)
            - log_1_minus_p
        )

    def np_prob(self, value):
        return np.exp(self.np_log_prob(value))

    def np_log_prob(self, value):
        eps = 1e-8
        cross_entropy = np.nan_to_num(
            value * np.log(self.probability)
            + (1.0 - value) * np.log(1 - self.probability),
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
        log_q = np.log(other.probability)
        log_1_minus_q = np.log1p(-other.probability)
        part2 = -(
            other._log_constant()
            + self.np_mean() * (log_q - log_1_minus_q)
            + log_1_minus_q
        )
        return part1 + part2


paddle.enable_static()


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'probability'),
    [
        (
            'multi-dim',
            parameterize.xrand((2, 3), min=0.1, max=0.9).astype("float32"),
        ),
    ],
)
class TestContinuousBernoulli(unittest.TestCase):
    def setUp(self):
        self._np_dist = ContinuousBernoulli_np(self.probability)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            probability = paddle.static.data(
                'probability', self.probability.shape, self.probability.dtype
            )
            dist = ContinuousBernoulli(probability)
            mean = dist.mean
            var = dist.variance
            entropy = dist.entropy()
            mini_samples = dist.sample(shape=())
            large_samples = dist.sample(shape=(1000,))
        fetch_list = [mean, var, entropy, mini_samples, large_samples]
        feed = {'probability': self.probability}

        executor.run(startup_program)
        [
            self.mean,
            self.var,
            self.entropy,
            self.mini_samples,
            self.large_samples,
        ] = executor.run(main_program, feed=feed, fetch_list=fetch_list)

    def test_mean(self):
        self.assertEqual(
            str(self.mean.dtype).split('.')[-1], self.probability.dtype
        )
        np.testing.assert_allclose(
            self.mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.probability.dtype)),
            atol=config.ATOL.get(str(self.probability.dtype)),
        )

    def test_variance(self):
        self.assertEqual(
            str(self.var.dtype).split('.')[-1], self.probability.dtype
        )
        np.testing.assert_allclose(
            self.var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.probability.dtype)),
            atol=config.ATOL.get(str(self.probability.dtype)),
        )

    def test_entropy(self):
        self.assertEqual(
            str(self.entropy.dtype).split('.')[-1], self.probability.dtype
        )
        np.testing.assert_allclose(
            self.entropy,
            self._np_entropy(),
            rtol=0.10,
            atol=0.10,
        )

    def test_sample(self):
        self.assertEqual(
            str(self.mini_samples.dtype).split('.')[-1], self.probability.dtype
        )
        sample_mean = self.large_samples.mean(axis=0)
        sample_variance = self.large_samples.var(axis=0)
        np.testing.assert_allclose(sample_mean, self.mean, atol=0, rtol=0.20)
        np.testing.assert_allclose(sample_variance, self.var, atol=0, rtol=0.20)

    def _np_variance(self):
        return self._np_dist.np_variance()

    def _np_mean(self):
        return self._np_dist.np_mean()

    def _np_entropy(self):
        return self._np_dist.np_entropy()


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'probability', 'value'),
    [
        (
            'value-broadcast-shape',
            parameterize.xrand((1,), min=0.1, max=0.9).astype("float32"),
            parameterize.xrand((2, 3), min=0.1, max=0.9).astype("float32"),
        ),
    ],
)
class TestContinuousBernoulliProbs(unittest.TestCase):
    def setUp(self):
        self._np_dist = ContinuousBernoulli_np(self.probability)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            probability = paddle.static.data(
                'probability', self.probability.shape, self.probability.dtype
            )
            value = paddle.static.data(
                'value', self.value.shape, self.value.dtype
            )
            dist = ContinuousBernoulli(probability)
            pmf = dist.prob(value)
        feed = {'probability': self.probability, 'value': self.value}
        fetch_list = [pmf]

        executor.run(startup_program)
        [self.pmf] = executor.run(
            main_program, feed=feed, fetch_list=fetch_list
        )

    def test_prob(self):
        np.testing.assert_allclose(
            self.pmf,
            self._np_dist.np_prob(self.value),
            rtol=config.RTOL.get(str(self.probability.dtype)),
            atol=config.ATOL.get(str(self.probability.dtype)),
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'p_1', 'p_2'),
    [
        (
            'multi-dim',
            parameterize.xrand((5,), min=0.1, max=0.9).astype("float32"),
            parameterize.xrand((5,), min=0.1, max=0.9).astype("float32"),
        ),
    ],
)
class TestContinuousBernoulliKL(unittest.TestCase):
    def setUp(self):
        self._np_dist1 = ContinuousBernoulli_np(self.p_1)
        self._np_dist2 = ContinuousBernoulli_np(self.p_2)

        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            p_1 = paddle.static.data('p_1', self.p_1.shape)
            p_2 = paddle.static.data('p_2', self.p_2.shape)
            dist1 = ContinuousBernoulli(p_1)
            dist2 = ContinuousBernoulli(p_2)
            kl_dist1_dist2 = dist1.kl_divergence(dist2)
        feed = {'p_1': self.p_1, 'p_2': self.p_2}
        fetch_list = [kl_dist1_dist2]

        executor.run(startup_program)
        [self.kl_dist1_dist2] = executor.run(
            main_program, feed=feed, fetch_list=fetch_list
        )

    def test_kl_divergence(self):
        kl0 = self.kl_dist1_dist2
        kl1 = self._np_dist1.np_kl_divergence(self._np_dist2)

        self.assertEqual(tuple(kl0.shape), self.p_1.shape)
        self.assertEqual(tuple(kl1.shape), self.p_1.shape)
        np.testing.assert_allclose(kl0, kl1, rtol=0.1, atol=0.1)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=3, exit=False)
