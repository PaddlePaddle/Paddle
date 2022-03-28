# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import fluid, tensor
import paddle
import paddle.fluid.dygraph as dg
import numpy as np
import unittest
from paddle.fluid.framework import _test_eager_guard


class ComplexKronTestCase(unittest.TestCase):
    def __init__(self, methodName='runTest', x=None, y=None):
        super(ComplexKronTestCase, self).__init__(methodName)
        self.x = x
        self.y = y

    def setUp(self):
        self.ref_result = np.kron(self.x, self.y)
        self._places = [paddle.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def runTest(self):
        for place in self._places:
            self.test_kron_api(place)
            self.test_eager(place)

    def test_kron_api(self, place):
        with dg.guard(place):
            x_var = dg.to_variable(self.x)
            y_var = dg.to_variable(self.y)
            out_var = paddle.kron(x_var, y_var)
            self.assertTrue(np.allclose(out_var.numpy(), self.ref_result))

    def test_eager(self, place):
        with _test_eager_guard():
            self.test_kron_api(place)


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    for dtype in ["float32", "float64"]:
        suite.addTest(
            ComplexKronTestCase(
                x=np.random.randn(2, 2).astype(dtype) + 1j * np.random.randn(
                    2, 2).astype(dtype),
                y=np.random.randn(3, 3).astype(dtype) + 1j * np.random.randn(
                    3, 3).astype(dtype)))
        suite.addTest(
            ComplexKronTestCase(
                x=np.random.randn(2, 2).astype(dtype),
                y=np.random.randn(3, 3).astype(dtype) + 1j * np.random.randn(
                    3, 3).astype(dtype)))
        suite.addTest(
            ComplexKronTestCase(
                x=np.random.randn(2, 2).astype(dtype) + 1j * np.random.randn(
                    2, 2).astype(dtype),
                y=np.random.randn(3, 3).astype(dtype)))

        suite.addTest(
            ComplexKronTestCase(
                x=np.random.randn(2, 2).astype(dtype) + 1j * np.random.randn(
                    2, 2).astype(dtype),
                y=np.random.randn(2, 2, 3).astype(dtype)))

    return suite


if __name__ == '__main__':
    unittest.main()
