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
from paddle import fluid
from paddle.fluid import core

paddle.enable_static()


class TestAmimGradCheck(unittest.TestCase):
    def setUp(self):
        self.init_test()

    def init_test(self):
        self.x_shape = [2]

    @prog_scope()
    def func(self, place):
        eps = 0.005
        dtype = np.float64
        typename = "float64"
        x = paddle.create_parameter(
            dtype=typename, shape=self.x_shape, name='x'
        )
        out = paddle.amin(x)

        x_arr = np.random.uniform(-1, 1, self.x_shape).astype(dtype)
        gradient_checker.grad_check(
            [x], out, x_init=[x_arr], place=place, eps=eps
        )

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


def TestAminGradCheckCase1(TestAmimGradCheck):
    def init_test(self):
        self.x_shape = [2, 3]


def TestAminGradCheckCase2(TestAmimGradCheck):
    def init_test(self):
        self.x_shape = [2, 4, 3]


if __name__ == "__main__":
    unittest.main()
