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

paddle.enable_static()


paddle.enable_static()


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
            np.array([1000]),
            parameterize.xrand((1,), dtype='float32', min=0, max=1),
        ),
        (
            'multi-dim',
            np.array([100]),
            parameterize.xrand((1, 3), dtype='float64', min=0, max=1),
        ),
    ],
)
class TestBinomial(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            probs = paddle.static.data(
                'probs', self.probs.shape, self.probs.dtype
            )
            total_count = paddle.static.data(
                'total_count', self.total_count.shape, self.total_count.dtype
            )
            dist = Binomial(total_count, probs)
            mean = dist.mean
            var = dist.variance
            entropy = dist.entropy()
            large_samples = dist.sample(shape=(1000,))
        fetch_list = [mean, var, entropy, large_samples]
        feed = {
            'probs': self.probs,
            'total_count': self.total_count,
        }

        executor.run(startup_program)
        [
            self.mean,
            self.var,
            self.entropy,
            self.large_samples,
        ] = executor.run(main_program, feed=feed, fetch_list=fetch_list)

    def test_mean(self):
        self.assertEqual(str(self.mean.dtype).split('.')[-1], self.probs.dtype)
        np.testing.assert_allclose(
            self.mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_variance(self):
        self.assertEqual(str(self.var.dtype).split('.')[-1], self.probs.dtype)
        np.testing.assert_allclose(
            self.var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_entropy(self):
        self.assertEqual(
            str(self.entropy.dtype).split('.')[-1], self.probs.dtype
        )
        np.testing.assert_allclose(
            self.entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )

    def test_sample(self):
        self.assertEqual(
            str(self.large_samples.dtype).split('.')[-1], self.probs.dtype
        )
        sample_mean = self.large_samples.mean(axis=0)
        sample_variance = self.large_samples.var(axis=0)
        np.testing.assert_allclose(sample_mean, self.mean, atol=0, rtol=0.20)
        np.testing.assert_allclose(sample_variance, self.var, atol=0, rtol=0.20)

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
            'zero-dim',
            np.array(10),
            np.array(0.6).astype('float64'),
            np.array([2.0, 3.0, 5.0]).astype('float64'),
        ),
        (
            'value-same-shape',
            np.array([10]).astype('int64'),
            np.array([0.2, 0.3, 0.5]).astype('float64'),
            np.array([2.0, 3.0, 5.0]).astype('float64'),
        ),
        (
            'value-broadcast-shape',
            np.array([10]),
            np.array([[0.3, 0.7], [0.5, 0.5]]),
            np.array([[[4.0, 6.0], [8.0, 2.0]], [[2.0, 4.0], [9.0, 7.0]]]),
        ),
    ],
)
class TestBinomialProbs(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            total_count = paddle.static.data(
                'total_count', self.total_count.shape, self.total_count.dtype
            )
            probs = paddle.static.data(
                'probs', self.probs.shape, self.probs.dtype
            )
            value = paddle.static.data(
                'value', self.value.shape, self.value.dtype
            )
            dist = Binomial(total_count, probs)
            pmf = dist.prob(value)
        feed = {
            'total_count': self.total_count,
            'probs': self.probs,
            'value': self.value,
        }
        fetch_list = [pmf]

        executor.run(startup_program)
        [self.pmf] = executor.run(
            main_program, feed=feed, fetch_list=fetch_list
        )

    def test_prob(self):
        np.testing.assert_allclose(
            self.pmf,
            scipy.stats.binom.pmf(self.value, self.total_count, self.probs),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'n_1', 'p_1', 'n_2', 'p_2'),
    [
        (
            'multi-dim-probability',
            np.array([32]),
            parameterize.xrand((1, 2), dtype='float64', min=0, max=1),
            np.array([32]),
            parameterize.xrand((1, 2), dtype='float64', min=0, max=1),
        ),
    ],
)
class TestBinomialKL(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            n_1 = paddle.static.data('n_1', self.n_1.shape, self.n_1.dtype)
            p_1 = paddle.static.data('p_1', self.p_1.shape, self.p_1.dtype)
            n_2 = paddle.static.data('n_2', self.n_2.shape, self.n_2.dtype)
            p_2 = paddle.static.data('p_2', self.p_2.shape, self.p_2.dtype)
            dist1 = Binomial(n_1, p_1)
            dist2 = Binomial(n_2, p_2)
            kl_dist1_dist2 = dist1.kl_divergence(dist2)
        feed = {
            'n_1': self.n_1,
            'p_1': self.p_1,
            'n_2': self.n_2,
            'p_2': self.p_2,
        }
        fetch_list = [kl_dist1_dist2]

        executor.run(startup_program)
        [self.kl_dist1_dist2] = executor.run(
            main_program, feed=feed, fetch_list=fetch_list
        )

    def test_kl_divergence(self):
        kl0 = self.kl_dist1_dist2
        kl1 = self.kl_divergence_scipy()

        self.assertEqual(tuple(kl0.shape), self.p_1.shape)
        self.assertEqual(tuple(kl1.shape), self.p_1.shape)
        np.testing.assert_allclose(
            kl0,
            kl1,
            rtol=config.RTOL.get(str(self.p_1.dtype)),
            atol=config.ATOL.get(str(self.p_1.dtype)),
        )

    def kl_divergence_scipy(self):
        support = np.arange(1 + self.n_1.max(), dtype=self.p_1.dtype)
        support = support.reshape((-1,) + (1,) * len(self.p_1.shape))
        log_prob_1 = scipy.stats.binom.logpmf(support, self.n_1, self.p_1)
        log_prob_2 = scipy.stats.binom.logpmf(support, self.n_2, self.p_2)
        return (np.exp(log_prob_1) * (log_prob_1 - log_prob_2)).sum(0)


if __name__ == '__main__':
    unittest.main()
