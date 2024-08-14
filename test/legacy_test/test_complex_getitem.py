#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class TestComplexGetitemLayer(unittest.TestCase):
    def setUp(self):
        self._places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            self._places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            self._places.append(base.CUDAPlace(0))

    def test_case1(self):
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0]

        for place in self._places:
            with dg.guard(place):
                x_var = paddle.to_tensor(x_np)
                x_var_slice = x_var[0]

            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case2(self):
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1]

        for place in self._places:
            with dg.guard(place):
                x_var = paddle.to_tensor(x_np)
                x_var_slice = x_var[0][1]

            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case3(self):
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1][2]

        for place in self._places:
            with dg.guard(place):
                x_var = paddle.to_tensor(x_np)
                x_var_slice = x_var[0][1][2]

            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case4(self):
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1][0:3]

        for place in self._places:
            with dg.guard(place):
                x_var = paddle.to_tensor(x_np)
                x_var_slice = x_var[0][1][0:3]

            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case5(self):
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1][0:4:2]

        for place in self._places:
            with dg.guard(place):
                x_var = paddle.to_tensor(x_np)
                x_var_slice = x_var[0][1][0:4:2]

            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)

    def test_case6(self):
        x_np = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
        x_np_slice = x_np[0][1:3][0:4:2]

        for place in self._places:
            with dg.guard(place):
                x_var = paddle.to_tensor(x_np)
                x_var_slice = x_var[0][1:3][0:4:2]

            np.testing.assert_allclose(x_var_slice.numpy(), x_np_slice)


if __name__ == '__main__':
    unittest.main()
