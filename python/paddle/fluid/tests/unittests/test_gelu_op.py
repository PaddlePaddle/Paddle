#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from scipy.special import erf
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg


def gelu(x, approximate):
    if approximate:
        y_ref = 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)


class TestGeluOp(unittest.TestCase):
    def _test_case1_cpu(self, approximate):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        y_ref = gelu(x, approximate)

        place = fluid.CPUPlace()
        with dg.guard(place) as g:
            x_var = dg.to_variable(x)
            y_var = fluid.layers.gelu(x_var, approximate)
            y_test = y_var.numpy()
        self.assertTrue(np.allclose(y_ref, y_test, rtol=1e-05, atol=1e-08))

    def _test_case1_gpu(self, approximate):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        y_ref = gelu(x, approximate)

        place = fluid.CUDAPlace(0)
        with dg.guard(place) as g:
            x_var = dg.to_variable(x)
            y_var = fluid.layers.gelu(x_var, approximate)
            y_test = y_var.numpy()
        self.assertTrue(np.allclose(y_ref, y_test, rtol=1e-05, atol=1e-08))

    def test_cases(self):
        for approximate in [True, False]:
            self._test_case1_cpu(approximate)
            if fluid.is_compiled_with_cuda():
                self._test_case1_gpu(approximate)


if __name__ == '__main__':
    unittest.main()
