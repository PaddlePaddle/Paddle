# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import scipy.stats
from op_test import OpTest

import paddle
from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "",
)
class TestStandardGammaOp(OpTest):
    def setUp(self):
        self.op_type = "standard_gamma"
        self.alpha = 0.1
        self.sample_shape = (100000, 2)
        self.init_dtype()

        self.inputs = {
            'x': np.broadcast_to(self.alpha, self.sample_shape).astype(
                self.dtype
            )
        }
        self.attrs = {}
        self.outputs = {'out': np.zeros(self.sample_shape, dtype=self.dtype)}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output_customized(self._hypothesis_testing)

    def _hypothesis_testing(self, outs):
        self.assertEqual(outs[0].shape, self.sample_shape)
        self.assertTrue(np.all(outs[0] > 0.0))
        self.assertLess(
            scipy.stats.kstest(
                outs[0][:, 0],
                scipy.stats.gamma(a=self.alpha).cdf,
            )[0],
            0.01,
        )

    def test_check_grad_normal(self):
        x = paddle.to_tensor(self.inputs['x'])
        x.stop_gradient = False
        y = paddle.standard_gamma(x)
        y.backward()
        grads = x.gradient()

        y = y.numpy()
        alpha = self.inputs['x']
        cdf = scipy.stats.gamma.cdf
        pdf = scipy.stats.gamma.pdf

        eps = 0.001 * alpha / (1.0 + alpha**0.5)
        cdf_alpha = (cdf(y, alpha + eps) - cdf(y, alpha - eps)) / (2 * eps)
        cdf_y = pdf(y, alpha)
        numeric_grads = -cdf_alpha / cdf_y

        np.testing.assert_allclose(grads, numeric_grads, rtol=0.05, atol=0)


class TestStandardGammaFP32Op(TestStandardGammaOp):
    def init_dtype(self):
        self.dtype = np.float32


if __name__ == '__main__':
    unittest.main()
