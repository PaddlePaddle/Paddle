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

paddle.enable_static()


@param.place(config.DEVICES)
@param.param_cls(
    (param.TEST_CASE_NAME, 'base', 'reinterpreted_batch_rank', 'alpha', 'beta'),
    [('base_beta', paddle.distribution.Beta, 1, np.random.rand(1, 2),
      np.random.rand(1, 2))])
class TestIndependent(unittest.TestCase):
    def setUp(self):
        value = np.random.rand(1)
        self.dtype = value.dtype
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            alpha = paddle.static.data('alpha', self.alpha.shape,
                                       self.alpha.dtype)
            beta = paddle.static.data('beta', self.beta.shape, self.beta.dtype)
            self.base = self.base(alpha, beta)
            t = paddle.distribution.Independent(self.base,
                                                self.reinterpreted_batch_rank)
            mean = t.mean
            variance = t.variance
            entropy = t.entropy()
            static_value = paddle.static.data('value', value.shape, value.dtype)
            log_prob = t.log_prob(static_value)

            base_mean = self.base.mean
            base_variance = self.base.variance
            base_entropy = self.base.entropy()
            base_log_prob = self.base.log_prob(static_value)

        fetch_list = [
            mean, variance, entropy, log_prob, base_mean, base_variance,
            base_entropy, base_log_prob
        ]
        exe.run(sp)
        [
            self.mean, self.variance, self.entropy, self.log_prob,
            self.base_mean, self.base_variance, self.base_entropy,
            self.base_log_prob
        ] = exe.run(
            mp,
            feed={'value': value,
                  'alpha': self.alpha,
                  'beta': self.beta},
            fetch_list=fetch_list)

    def test_mean(self):
        np.testing.assert_allclose(
            self.mean,
            self.base_mean,
            rtol=config.RTOL.get(str(self.dtype)),
            atol=config.ATOL.get(str(self.dtype)))

    def test_variance(self):
        np.testing.assert_allclose(
            self.variance,
            self.base_variance,
            rtol=config.RTOL.get(str(self.dtype)),
            atol=config.ATOL.get(str(self.dtype)))

    def test_entropy(self):
        np.testing.assert_allclose(
            self._np_sum_rightmost(self.base_entropy,
                                   self.reinterpreted_batch_rank),
            self.entropy,
            rtol=config.RTOL.get(str(self.dtype)),
            atol=config.ATOL.get(str(self.dtype)))

    def _np_sum_rightmost(self, value, n):
        return np.sum(value, tuple(range(-n, 0))) if n > 0 else value

    def test_log_prob(self):
        np.testing.assert_allclose(
            self._np_sum_rightmost(self.base_log_prob,
                                   self.reinterpreted_batch_rank),
            self.log_prob,
            rtol=config.RTOL.get(str(self.dtype)),
            atol=config.ATOL.get(str(self.dtype)))


if __name__ == '__main__':
    unittest.main()
