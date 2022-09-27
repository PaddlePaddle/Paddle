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

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from op_test import OpTest

paddle.enable_static()


class TestRad2degAPI(unittest.TestCase):

    def setUp(self):
        self.x_dtype = 'float64'
        self.x_np = np.array([3.142, -3.142, 6.283, -6.283, 1.570,
                              -1.570]).astype(np.float64)
        self.x_shape = [6]
        self.out_np = np.rad2deg(self.x_np)

    def test_static_graph(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(startup_program, train_program):
            x = fluid.data(name='input', dtype=self.x_dtype, shape=self.x_shape)
            out = paddle.rad2deg(x)

            place = fluid.CUDAPlace(
                0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
            exe = fluid.Executor(place)
            res = exe.run(fluid.default_main_program(),
                          feed={'input': self.x_np},
                          fetch_list=[out])
            self.assertTrue((np.array(out[0]) == self.out_np).all())

    def test_dygraph(self):
        paddle.disable_static()
        x1 = paddle.to_tensor([3.142, -3.142, 6.283, -6.283, 1.570, -1.570])
        result1 = paddle.rad2deg(x1)
        np.testing.assert_allclose(self.out_np, result1.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestRad2degAPI2(TestRad2degAPI):

    def setUp(self):
        self.x_np = np.pi / 2
        self.x_shape = [1]
        self.out_np = 90
        self.x_dtype = 'float32'

    def test_dygraph(self):
        paddle.disable_static()

        x2 = paddle.to_tensor(np.pi / 2)
        result2 = paddle.rad2deg(x2)
        np.testing.assert_allclose(90, result2.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestRad2degAPI3(TestRad2degAPI):
    # Test input data type is int
    def setUp(self):
        self.x_np = 1
        self.x_shape = [1]
        self.out_np = 180 / np.pi
        self.x_dtype = 'int64'

    def test_dygraph(self):
        paddle.disable_static()

        x2 = paddle.to_tensor(1)
        result2 = paddle.rad2deg(x2)
        np.testing.assert_allclose(180 / np.pi, result2.numpy(), rtol=1e-05)

        paddle.enable_static()
