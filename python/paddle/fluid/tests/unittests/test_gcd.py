#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from op_test import OpTest

paddle.enable_static()


class TestGcdAPI(unittest.TestCase):
    def setUp(self):
        self.x_np = 12
        self.y_np = 20
        self.x_shape = [1]
        self.y_shape = [1]

    def test_static_graph(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(startup_program, train_program):
            x = fluid.data(name='input1', dtype='int32', shape=self.x_shape)
            y = fluid.data(name='input2', dtype='int32', shape=self.y_shape)
            out = paddle.gcd(x, y)

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            res = exe.run(fluid.default_main_program(),
                          feed={'input1': self.x_np,
                                'input2': self.y_np},
                          fetch_list=[out])
            self.assertTrue((np.array(res[0]) == np.gcd(self.x_np, self.y_np)
                             ).all())

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        result = paddle.gcd(x, y)
        self.assertEqual(
            np.allclose(np.gcd(self.x_np, self.y_np), result.numpy()), True)

        paddle.enable_static()


class TestGcdAPI2(TestGcdAPI):
    def setUp(self):
        self.x_np = np.arange(6).astype(np.int32)
        self.y_np = np.array([20]).astype(np.int32)
        self.x_shape = [6]
        self.y_shape = [1]


class TestGcdAPI3(TestGcdAPI):
    def setUp(self):
        self.x_np = 0
        self.y_np = 20
        self.x_shape = [1]
        self.y_shape = [1]


class TestGcdAPI4(TestGcdAPI):
    def setUp(self):
        self.x_np = 0
        self.y_np = 0
        self.x_shape = [1]
        self.y_shape = [1]


class TestGcdAPI5(TestGcdAPI):
    def setUp(self):
        self.x_np = 12
        self.y_np = -20
        self.x_shape = [1]
        self.y_shape = [1]
