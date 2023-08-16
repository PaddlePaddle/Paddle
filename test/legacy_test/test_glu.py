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

import unittest

import numpy as np

import paddle
import paddle.base.dygraph as dg
from paddle import base
from paddle.nn import functional as F


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def glu(x, dim=-1):
    a, b = np.split(x, 2, axis=dim)
    out = a * sigmoid(b)
    return out


class TestGLUV2(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(5, 20)
        self.dim = -1
        self.out = glu(self.x, self.dim)

    def check_identity(self, place):
        with dg.guard(place):
            x_var = paddle.to_tensor(self.x)
            y_var = F.glu(x_var, self.dim)
            y_np = y_var.numpy()

        np.testing.assert_allclose(y_np, self.out)

    def test_case(self):
        self.check_identity(base.CPUPlace())
        if base.is_compiled_with_cuda():
            self.check_identity(base.CUDAPlace(0))


class TestGlu(unittest.TestCase):
    def glu_axis_size(self):
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[1, 2, 3], dtype='float32')
        paddle.nn.functional.glu(x, axis=256)

    def test_errors(self):
        self.assertRaises(ValueError, self.glu_axis_size)


if __name__ == '__main__':
    unittest.main()
