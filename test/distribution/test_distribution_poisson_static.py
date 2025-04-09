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

paddle.enable_static()


paddle.enable_static()


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'rate'),
    [
        ('zero-dim', np.array(1000.0).astype('float32')),
        ('one-dim', np.array([1000.0]).astype('float32')),
        (
            'multi-dim',
            parameterize.xrand((2,), min=1, max=20)
            .astype('int32')
            .astype('float64'),
        ),
    ],
)
class TestPoisson(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            rate = paddle.static.data('rate', self.rate.shape, self.rate.dtype)
            dist = Poisson(rate)
            mean = dist.mean
            var = dist.variance
            entropy = dist.entropy()
            mini_samples = dist.sample(shape=())
            large_samples = dist.sample(shape=(1000,))
        fetch_list = [mean, var, entropy, mini_samples, large_samples]
        feed = {'rate': self.rate}

        executor.run(startup_program)
        [
            self.mean,
            self.var,
            self.entropy,
            self.mini_samples,
            self.large_samples,
        ] = executor.run(main_program, feed=feed, fetch_list=fetch_list)

    def test_mean(self):
        self.assertEqual(str(self.mean.dtype).split('.')[-1], self.rate.dtype)
        np.testing.assert_allclose(
            self.mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.rate.dtype)),
            atol=config.ATOL.get(str(self.rate.dtype)),
        )

    def test_variance(self):
        self.assertEqual(str(self.var.dtype).split('.')[-1], self.rate.dtype)
        np.testing.assert_allclose(
            self.var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.rate.dtype)),
            atol=config.ATOL.get(str(self.rate.dtype)),
        )

    def test_entropy(self):
        self.assertEqual(
            str(self.entropy.dtype).split('.')[-1], self.rate.dtype
        )
        np.testing.assert_allclose(
            self.entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(self.rate.dtype)),
            atol=config.ATOL.get(str(self.rate.dtype)),
        )

    def test_sample(self):
        self.assertEqual(
            str(self.mini_samples.dtype).split('.')[-1], self.rate.dtype
        )
        sample_mean = self.large_samples.mean(axis=0)
        sample_variance = self.large_samples.var(axis=0)
        np.testing.assert_allclose(sample_mean, self.mean, atol=0, rtol=0.20)
        np.testing.assert_allclose(sample_variance, self.var, atol=0, rtol=0.20)

    def _np_variance(self):
        return self.rate

    def _np_mean(self):
        return self.rate

    def _np_entropy(self):
        return scipy.stats.poisson.entropy(self.rate)


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
            np.array([2.0, 3.0]).astype('float64'),
        ),
    ],
)
class TestPoissonProbs(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            rate = paddle.static.data('rate', self.rate.shape, self.rate.dtype)
            value = paddle.static.data(
                'value', self.value.shape, self.value.dtype
            )
            dist = Poisson(rate)
            pmf = dist.prob(value)
        feed = {'rate': self.rate, 'value': self.value}
        fetch_list = [pmf]

        executor.run(startup_program)
        [self.pmf] = executor.run(
            main_program, feed=feed, fetch_list=fetch_list
        )

    def test_prob(self):
        np.testing.assert_allclose(
            self.pmf,
            scipy.stats.poisson.pmf(self.value, self.rate),
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
            .astype('float32')
            .reshape([]),
            parameterize.xrand((1,), min=1, max=20)
            .astype('int32')
            .astype('float32')
            .reshape([]),
        ),
        (
            'multi-dim',
            parameterize.xrand((2, 3), min=1, max=20)
            .astype('int32')
            .astype('float32'),
            parameterize.xrand((2, 3), min=1, max=20)
            .astype('int32')
            .astype('float32'),
        ),
    ],
)
class TestPoissonKL(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            rate_1 = paddle.static.data('rate_1', self.rate_1.shape)
            rate_2 = paddle.static.data('rate_2', self.rate_2.shape)
            dist1 = Poisson(rate_1)
            dist2 = Poisson(rate_2)
            kl_dist1_dist2 = dist1.kl_divergence(dist2)
        feed = {'rate_1': self.rate_1, 'rate_2': self.rate_2}
        fetch_list = [kl_dist1_dist2]

        executor.run(startup_program)
        [self.kl_dist1_dist2] = executor.run(
            main_program, feed=feed, fetch_list=fetch_list
        )

    def test_kl_divergence(self):
        kl0 = self.kl_dist1_dist2
        kl1 = self.kl_divergence_scipy()

        self.assertEqual(tuple(kl0.shape), self.rate_1.shape)
        self.assertEqual(tuple(kl1.shape), self.rate_1.shape)
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
