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

import unittest

import gradient_checker
import numpy as np
from decorator_helper import prog_scope

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()


class TestMatmulDoubleGradCheck(unittest.TestCase):
    def setUp(self):
        self.init_test()

    def init_test(self):
        self.x_shape = [2]
        self.y_shape = [2]
        self.transpose_x = False
        self.transpose_y = False

    @prog_scope()
    def func(self, place):
        eps = 0.005
        dtype = np.float64
        typename = "float64"
        x = paddle.create_parameter(
            dtype=typename, shape=self.x_shape, name='x'
        )
        y = paddle.create_parameter(
            dtype=typename, shape=self.y_shape, name='y'
        )
        out = paddle.matmul(
            x, y, self.transpose_x, self.transpose_y, name='out'
        )

        x_arr = np.random.uniform(-1, 1, self.x_shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, self.y_shape).astype(dtype)
        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )

    def test_grad(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


def TestMatmulDoubleGradCheckCase1(TestMatmulDoubleGradCheck):
    def init_test(self):
        self.x_shape = [2, 3]
        self.y_shape = [3, 2]
        self.transpose_x = True
        self.transpose_y = True


def TestMatmulDoubleGradCheckCase2(TestMatmulDoubleGradCheck):
    def init_test(self):
        self.x_shape = [2, 4, 3]
        self.y_shape = [2, 4, 5]
        self.transpose_x = True
        self.transpose_y = False


def TestMatmulDoubleGradCheckCase3(TestMatmulDoubleGradCheck):
    def init_test(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 3, 5]
        self.transpose_x = False
        self.transpose_y = True


def TestMatmulDoubleGradCheckCase4(TestMatmulDoubleGradCheck):
    def init_test(self):
        self.x_shape = [2, 3, 4]
        self.y_shape = [4, 3]
        self.transpose_x = False
        self.transpose_y = False


if __name__ == "__main__":
    unittest.main()
