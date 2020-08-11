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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid.core as core


class ApiMaximumTest(unittest.TestCase):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

        self.input_x = np.random.rand(10, 15).astype("float32")
        self.input_y = np.random.rand(10, 15).astype("float32")
        self.input_z = np.random.rand(15).astype("float32")

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data_x = paddle.nn.data("x", shape=[10, 15], dtype="float32")
            data_y = paddle.nn.data("y", shape=[10, 15], dtype="float32")
            result_max = paddle.maximum(data_x, data_y)
            exe = paddle.static.Executor(self.place)
            res, = exe.run(feed={"x": self.input_x,
                                 "y": self.input_y},
                           fetch_list=[result_max])
        self.assertEqual((res == np.maximum(self.input_x, self.input_y)).all(),
                         True)

        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data_x = paddle.nn.data("x", shape=[10, 15], dtype="float32")
            data_z = paddle.nn.data("z", shape=[15], dtype="float32")
            result_max = paddle.maximum(data_x, data_z, axis=1)
            exe = paddle.static.Executor(self.place)
            res, = exe.run(feed={"x": self.input_x,
                                 "z": self.input_z},
                           fetch_list=[result_max])
        self.assertEqual((res == np.maximum(self.input_x, self.input_z)).all(),
                         True)

    def test_dynamic_api(self):
        paddle.disable_static()
        np_x = np.array([10, 10]).astype('float64')
        x = paddle.to_variable(self.input_x)
        y = paddle.to_variable(self.input_y)
        z = paddle.maximum(x, y)
        np_z = z.numpy()
        z_expected = np.array(np.maximum(self.input_x, self.input_y))
        self.assertEqual((np_z == z_expected).all(), True)

    def test_broadcast_axis(self):
        paddle.disable_static()
        np_x = np.random.rand(5, 4, 3, 2).astype("float64")
        np_y = np.random.rand(4, 3).astype("float64")

        x = paddle.to_variable(self.input_x)
        y = paddle.to_variable(self.input_y)
        result_1 = paddle.maximum(x, y, axis=1)
        result_2 = paddle.maximum(x, y, axis=-2)
        self.assertEqual((result_1.numpy() == result_2.numpy()).all(), True)
