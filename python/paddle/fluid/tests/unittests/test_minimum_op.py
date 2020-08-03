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


class ApiMinimumTest(unittest.TestCase):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

        self.input_x = np.random.rand(10, 15).astype("float32")
        self.input_y = np.random.rand(10, 15).astype("float32")
        self.input_z = np.random.rand(15).astype("float32")

    def test_api(self):
        with paddle.program_guard(paddle.Program(), paddle.Program()):
            data_x = paddle.nn.data("x", shape=[10, 15], dtype="float32")
            data_y = paddle.nn.data("y", shape=[10, 15], dtype="float32")
            result_min = paddle.minimum(data_x, data_y)
            exe = paddle.Executor(self.place)
            res, = exe.run(feed={"x": self.input_x,
                                 "y": self.input_y},
                           fetch_list=[result_min])
        self.assertEqual((res == np.minimum(self.input_x, self.input_y)).all(),
                         True)

        with paddle.program_guard(paddle.Program(), paddle.Program()):
            data_x = paddle.nn.data("x", shape=[10, 15], dtype="float32")
            data_z = paddle.nn.data("z", shape=[15], dtype="float32")
            result_min = paddle.minimum(data_x, data_z, axis=1)
            exe = paddle.Executor(self.place)
            res, = exe.run(feed={"x": self.input_x,
                                 "z": self.input_z},
                           fetch_list=[result_min])
        self.assertEqual((res == np.minimum(self.input_x, self.input_z)).all(),
                         True)

    def test_imperative_api(self):
        with paddle.imperative.guard(self.place):
            np_x = np.array([10, 10]).astype('float64')
            x = paddle.imperative.to_variable(self.input_x)
            y = paddle.imperative.to_variable(self.input_y)
            z = paddle.minimum(x, y)
            np_z = z.numpy()
            z_expected = np.array(np.minimum(self.input_x, self.input_y))
            self.assertEqual((np_z == z_expected).all(), True)
