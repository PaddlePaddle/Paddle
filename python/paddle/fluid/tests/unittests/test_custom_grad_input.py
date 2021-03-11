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

from __future__ import print_function

import unittest
import numpy as np

import paddle
import paddle.fluid.dygraph as dg
from op_test import OpTest


class TestBackward(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_all_positive(self):
        for dtype in self._dtypes:
            x = np.random.random([2, 100]).astype(dtype)
            y = np.random.random([100, 2]).astype(dtype)
            z = np.matmul(x, y)
            grad = np.random.random(z.shape).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    x_tensor = paddle.to_tensor(x, stop_gradient=False)
                    y_tensor = paddle.to_tensor(y)
                    z_tensor = paddle.matmul(x_tensor, y_tensor)

                    grad_tensor = paddle.to_tensor(grad)
                    z_tensor.backward(grad_tensor=grad_tensor)

                    x_grad = np.matmul(grad, y.T)

                    self.assertTrue(np.allclose(x_grad, x_tensor.grad))


if __name__ == '__main__':
    unittest.main()
