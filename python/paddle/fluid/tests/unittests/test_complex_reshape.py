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

import paddle.fluid as fluid
import paddle
import paddle.fluid.dygraph as dg
import numpy as np
import unittest
from paddle.fluid.framework import _test_eager_guard


class TestComplexReshape(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_shape_norm_dims(self):
        for dtype in self._dtypes:
            x_np = np.random.randn(
                2, 3, 4).astype(dtype) + 1j * np.random.randn(2, 3,
                                                              4).astype(dtype)
            shape = (2, -1)
            for place in self._places:
                with dg.guard(place):
                    x_var = dg.to_variable(x_np)
                    y_var = paddle.reshape(x_var, shape)
                    y_np = y_var.numpy()
                    self.assertTrue(np.allclose(np.reshape(x_np, shape), y_np))

    def test_shape_omit_dims(self):
        for dtype in self._dtypes:
            x_np = np.random.randn(
                2, 3, 4).astype(dtype) + 1j * np.random.randn(2, 3,
                                                              4).astype(dtype)
            shape = (0, -1)
            shape_ = (2, 12)
            for place in self._places:
                with dg.guard(place):
                    x_var = dg.to_variable(x_np)
                    y_var = paddle.reshape(x_var, shape)
                    y_np = y_var.numpy()
                    self.assertTrue(np.allclose(np.reshape(x_np, shape_), y_np))

    def test_eager(self):
        with _test_eager_guard():
            self.test_shape_norm_dims()
            self.test_shape_omit_dims()


if __name__ == "__main__":
    unittest.main()
