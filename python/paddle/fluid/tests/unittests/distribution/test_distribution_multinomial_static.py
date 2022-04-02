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
import paddle
import scipy.stats

import config
import parameterize

paddle.enable_static()


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'total_count', 'probs'), [
        ('one-dim', 5, parameterize.xrand((3, ))),
        ('multi-dim', 9, parameterize.xrand((2, 3))),
        ('prob-sum-one', 5, np.array([0.5, 0.2, 0.3])),
        ('prob-sum-non-one', 5, np.array([2., 3., 5.])),
    ])
class TestMultinomial(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            probs = paddle.static.data('probs', self.probs.shape,
                                       self.probs.dtype)
            dist = paddle.distribution.Multinomial(self.total_count, probs)
            mean = dist.mean
            var = dist.variance
            entropy = dist.entropy()
            mini_samples = dist.sample(shape=(6, ))
            large_samples = dist.sample(shape=(5000, ))
        fetch_list = [mean, var, entropy, mini_samples, large_samples]
        feed = {'probs': self.probs}

        executor.run(startup_program)
        [
            self.mean, self.var, self.entropy, self.mini_samples,
            self.large_samples
        ] = executor.run(main_program, feed=feed, fetch_list=fetch_list)

    def test_mean(self):
        self.assertEqual(str(self.mean.dtype).split('.')[-1], self.probs.dtype)
        np.testing.assert_allclose(
            self.mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))

    def test_variance(self):
        self.assertEqual(str(self.var.dtype).split('.')[-1], self.probs.dtype)
        np.testing.assert_allclose(
            self.var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))

    def test_entropy(self):
        self.assertEqual(
            str(self.entropy.dtype).split('.')[-1], self.probs.dtype)
        np.testing.assert_allclose(
            self.entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))

    def test_sample(self):
        self.assertEqual(
            str(self.mini_samples.dtype).split('.')[-1], self.probs.dtype)
        self.assertTrue(np.all(self.mini_samples.sum(-1) == self.total_count))

        sample_mean = self.large_samples.mean(axis=0)
        np.testing.assert_allclose(sample_mean, self.mean, atol=0, rtol=0.20)

    def _np_variance(self):
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return self.total_count * probs * (1 - probs)

    def _np_mean(self):
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return self.total_count * probs

    def _np_entropy(self):
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return scipy.stats.multinomial.entropy(self.total_count, probs)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'total_count', 'probs', 'value'),
    [
        ('value-float', 5, np.array([0.2, 0.3, 0.5]), np.array([1., 1., 3.])),
        ('value-int', 5, np.array([0.2, 0.3, 0.5]), np.array([2, 2, 1])),
        ('value-multi-dim', 5, np.array([[0.3, 0.7], [0.5, 0.5]]),
         np.array([[1., 4.], [2., 3.]])),
        # ('value-sum-non-n', 10, np.array([0.5, 0.2, 0.3]), np.array([4,5,2])),
    ])
class TestMultinomialPmf(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            probs = paddle.static.data('probs', self.probs.shape,
                                       self.probs.dtype)
            value = paddle.static.data('value', self.value.shape,
                                       self.value.dtype)
            dist = paddle.distribution.Multinomial(self.total_count, probs)
            pmf = dist.prob(value)
        feed = {'probs': self.probs, 'value': self.value}
        fetch_list = [pmf]

        executor.run(startup_program)
        [self.pmf] = executor.run(main_program,
                                  feed=feed,
                                  fetch_list=fetch_list)

    def test_prob(self):
        np.testing.assert_allclose(
            self.pmf,
            scipy.stats.multinomial.pmf(self.value, self.total_count,
                                        self.probs),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'total_count', 'probs'), [
        ('total_count_le_one', 0, np.array([0.3, 0.7])),
        ('total_count_float', np.array([0.3, 0.7])),
        ('probs_zero_dim', np.array(0)),
    ])
class TestMultinomialException(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        self.main_program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            probs = paddle.static.data('probs', self.probs.shape,
                                       self.probs.dtype)
            dist = paddle.distribution.Multinomial(self.total_count, probs)
        self.feed = {'probs': self.probs}

        executor.run(startup_program)

    def TestInit(self):
        with self.assertRaises(ValueError):
            self.executor.run(self.main_program, feed=self.feed, fetch=[])


if __name__ == '__main__':
    unittest.main()
