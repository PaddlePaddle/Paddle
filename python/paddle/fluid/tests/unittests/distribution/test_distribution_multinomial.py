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


@config.place(config.DEVICES)
@config.parameterize((config.TEST_CASE_NAME, 'total_count', 'probs'), [
    ('one-dim', 100, config.xrand((3, ))),
    ('multi-dim', 100, config.xrand((10, 20, 30))),
    ('prob-sum-one', 10, np.array([0.5, 0.2, 0.3])),
    ('prob-sum-non-one', 10, np.array([2, 3, 5])),
])
class TestMultinomial(unittest.TestCase):
    def setUp(self):
        self._dist = paddle.distribution.Multinomial(
            total_count=self.total_count, probs=paddle.to_tensor(self.probs))

    def test_mean(self):
        np.testing.assert_allclose(
            self._dist.mean,
            scipy.stats.multinomial.mean(self.total_count, self.probs),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))

    def test_variance(self):
        np.testing.assert_allclose(
            self._dist.variance,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))

    def test_entropy(self):
        np.testing.assert_allclose(
            self._dist.entropy(),
            scipy.stats.multinomial.entropy(self.total_count, self.probs),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))

    def _np_variance(self):
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return self.total_count * probs * (1 - probs)


@config.place(config.DEVICES)
@config.parameterize(
    (config.TEST_CASE_NAME, 'total_count', 'probs', 'value'),
    [
        ('value-float', 10, np.array([0.2, 0.3, 0.5]), np.array([2., 3., 5.])),
        ('value-int', 10, np.array([0.2, 0.3, 0.5]), np.array([2, 3, 5])),
        ('value-multi-dim', 10, np.array([[0.3, 0.7], [0.5, 0.5]]),
         np.array([[4., 6], [8, 2]])),
        # ('value-sum-non-n', 10, np.array([0.5, 0.2, 0.3]), np.array([4,5,2])),
    ])
class TestMultinomialPmf(unittest.TestCase):
    def setUp(self):
        self._dist = paddle.distribution.Multinomial(
            total_count=self.total_count, probs=paddle.to_tensor(self.probs))

    def test_prob(self):
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            scipy.stats.multinomial.pmf(self.value, self.total_count,
                                        self.probs),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))

    @config.place(config.DEVICES)
    @config.parameterize((config.TEST_CASE_NAME, 'total_count', 'probs'), [
        ('total_count_le_one', 0, np.array([0.3, 0.7])),
        ('total_count_float', np.array([0.3, 0.7])),
        ('probs_zero_dim', np.array(0)),
    ])
    class TestMultinomialException(unittest.TestCase):
        def TestInit(self):
            with self.assertRaises(ValueError):
                paddle.distribution.Multinomial(self.total_count,
                                                paddle.to_tensor(self.probs))


if __name__ == '__main__':
    unittest.main()
