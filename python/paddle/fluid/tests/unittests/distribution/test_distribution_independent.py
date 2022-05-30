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
import numbers
import unittest

import numpy as np
import paddle
import scipy.stats

import config
import parameterize as param


@param.place(config.DEVICES)
@param.param_cls((param.TEST_CASE_NAME, 'base', 'reinterpreted_batch_rank'),
                 [('base_beta', paddle.distribution.Beta(
                     paddle.rand([1, 2]), paddle.rand([1, 2])), 1)])
class TestIndependent(unittest.TestCase):
    def setUp(self):
        self._t = paddle.distribution.Independent(self.base,
                                                  self.reinterpreted_batch_rank)

    def test_mean(self):
        np.testing.assert_allclose(
            self.base.mean,
            self._t.mean,
            rtol=config.RTOL.get(str(self.base.alpha.numpy().dtype)),
            atol=config.ATOL.get(str(self.base.alpha.numpy().dtype)))

    def test_variance(self):
        np.testing.assert_allclose(
            self.base.variance,
            self._t.variance,
            rtol=config.RTOL.get(str(self.base.alpha.numpy().dtype)),
            atol=config.ATOL.get(str(self.base.alpha.numpy().dtype)))

    def test_entropy(self):
        np.testing.assert_allclose(
            self._np_sum_rightmost(self.base.entropy().numpy(),
                                   self.reinterpreted_batch_rank),
            self._t.entropy(),
            rtol=config.RTOL.get(str(self.base.alpha.numpy().dtype)),
            atol=config.ATOL.get(str(self.base.alpha.numpy().dtype)))

    def _np_sum_rightmost(self, value, n):
        return np.sum(value, tuple(range(-n, 0))) if n > 0 else value

    def test_log_prob(self):
        value = np.random.rand(1)
        np.testing.assert_allclose(
            self._np_sum_rightmost(
                self.base.log_prob(paddle.to_tensor(value)).numpy(),
                self.reinterpreted_batch_rank),
            self._t.log_prob(paddle.to_tensor(value)).numpy(),
            rtol=config.RTOL.get(str(self.base.alpha.numpy().dtype)),
            atol=config.ATOL.get(str(self.base.alpha.numpy().dtype)))

    # TODO(cxxly): Add Kolmogorov-Smirnov test for sample result.
    def test_sample(self):
        shape = (5, 10, 8)
        expected_shape = (5, 10, 8, 1, 2)
        data = self._t.sample(shape)
        self.assertEqual(tuple(data.shape), expected_shape)
        self.assertEqual(data.dtype, self.base.alpha.dtype)


@param.place(config.DEVICES)
@param.param_cls(
    (param.TEST_CASE_NAME, 'base', 'reinterpreted_batch_rank',
     'expected_exception'),
    [('base_not_transform', '', 1, TypeError),
     ('rank_less_than_zero', paddle.distribution.Transform(), -1, ValueError)])
class TestIndependentException(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(self.expected_exception):
            paddle.distribution.IndependentTransform(
                self.base, self.reinterpreted_batch_rank)


if __name__ == '__main__':
    unittest.main()
