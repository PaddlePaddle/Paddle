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
from config import ATOL, DEVICES, RTOL
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand


@place(DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'alpha', 'beta'),
                  [('test-scale', 1.0, 2.0), ('test-tensor', xrand(), xrand()),
                   ('test-broadcast', xrand((2, 1)), xrand((2, 5)))])
class TestBeta(unittest.TestCase):
    def setUp(self):
        # scale no need convert to tensor for scale input unittest
        alpha, beta = self.alpha, self.beta
        if not isinstance(self.alpha, numbers.Real):
            alpha = paddle.to_tensor(self.alpha)
        if not isinstance(self.beta, numbers.Real):
            beta = paddle.to_tensor(self.beta)

        self._paddle_beta = paddle.distribution.Beta(alpha, beta)

    def test_mean(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_beta.mean,
                scipy.stats.beta.mean(self.alpha, self.beta),
                rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_variance(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_beta.variance,
                scipy.stats.beta.var(self.alpha, self.beta),
                rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_prob(self):
        value = [np.random.rand(*self._paddle_beta.alpha.shape)]

        for v in value:
            with paddle.fluid.dygraph.guard(self.place):
                np.testing.assert_allclose(
                    self._paddle_beta.prob(paddle.to_tensor(v)),
                    scipy.stats.beta.pdf(v, self.alpha, self.beta),
                    rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)),
                    atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_log_prob(self):
        value = [np.random.rand(*self._paddle_beta.alpha.shape)]

        for v in value:
            with paddle.fluid.dygraph.guard(self.place):
                np.testing.assert_allclose(
                    self._paddle_beta.log_prob(paddle.to_tensor(v)),
                    scipy.stats.beta.logpdf(v, self.alpha, self.beta),
                    rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)),
                    atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_entropy(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_beta.entropy(),
                scipy.stats.beta.entropy(self.alpha, self.beta),
                rtol=RTOL.get(str(self._paddle_beta.alpha.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_beta.alpha.numpy().dtype)))

    def test_sample_shape(self):
        cases = [
            {
                'input': [],
                'expect': [] + paddle.squeeze(self._paddle_beta.alpha).shape
            },
            {
                'input': [2, 3],
                'expect': [2, 3] + paddle.squeeze(self._paddle_beta.alpha).shape
            },
        ]
        for case in cases:
            self.assertTrue(
                self._paddle_beta.sample(case.get('input')).shape ==
                case.get('expect'))


if __name__ == '__main__':
    unittest.main()
