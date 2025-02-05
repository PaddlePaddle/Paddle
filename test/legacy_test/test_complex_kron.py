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

import os
import unittest

import numpy as np

import paddle
import paddle.base.dygraph as dg
from paddle import base


class ComplexKronTestCase(unittest.TestCase):
    def __init__(self, methodName='runTest', x=None, y=None):
        super().__init__(methodName)
        self.x = x
        self.y = y

    def setUp(self):
        self.ref_result = np.kron(self.x, self.y)
        self._places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            self._places.append(paddle.CPUPlace())
        if base.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def runTest(self):
        for place in self._places:
            self.test_kron_api(place)

    def test_kron_api(self, place):
        with dg.guard(place):
            x_var = paddle.to_tensor(self.x)
            y_var = paddle.to_tensor(self.y)
            out_var = paddle.kron(x_var, y_var)
            np.testing.assert_allclose(
                out_var.numpy(), self.ref_result, rtol=1e-05
            )


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    for dtype in ["float32", "float64"]:
        suite.addTest(
            ComplexKronTestCase(
                x=np.random.randn(2, 2).astype(dtype)
                + 1j * np.random.randn(2, 2).astype(dtype),
                y=np.random.randn(3, 3).astype(dtype)
                + 1j * np.random.randn(3, 3).astype(dtype),
            )
        )
        suite.addTest(
            ComplexKronTestCase(
                x=np.random.randn(2, 2).astype(dtype),
                y=np.random.randn(3, 3).astype(dtype)
                + 1j * np.random.randn(3, 3).astype(dtype),
            )
        )
        suite.addTest(
            ComplexKronTestCase(
                x=np.random.randn(2, 2).astype(dtype)
                + 1j * np.random.randn(2, 2).astype(dtype),
                y=np.random.randn(3, 3).astype(dtype),
            )
        )

        suite.addTest(
            ComplexKronTestCase(
                x=np.random.randn(2, 2).astype(dtype)
                + 1j * np.random.randn(2, 2).astype(dtype),
                y=np.random.randn(2, 2, 3).astype(dtype),
            )
        )

    return suite


if __name__ == '__main__':
    unittest.main()
