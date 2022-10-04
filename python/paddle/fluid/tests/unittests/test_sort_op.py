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
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
import six
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard


class TestSortOnCPU(unittest.TestCase):

    def setUp(self):
        self.place = core.CPUPlace()

    def test_api_0(self):
        with fluid.program_guard(fluid.Program()):
            input = fluid.data(name="input", shape=[2, 3, 4], dtype="float32")
            output = paddle.sort(x=input)
            exe = fluid.Executor(self.place)
            data = np.array([[[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
                             [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]]],
                            dtype='float32')
            result, = exe.run(feed={'input': data}, fetch_list=[output])
            np_result = np.sort(result)
            self.assertEqual((result == np_result).all(), True)

    def test_api_1(self):
        with fluid.program_guard(fluid.Program()):
            input = fluid.data(name="input", shape=[2, 3, 4], dtype="float32")
            output = paddle.sort(x=input, axis=1)
            exe = fluid.Executor(self.place)
            data = np.array([[[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
                             [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]]],
                            dtype='float32')
            result, = exe.run(feed={'input': data}, fetch_list=[output])
            np_result = np.sort(result, axis=1)
            self.assertEqual((result == np_result).all(), True)


class TestSortOnGPU(TestSortOnCPU):

    def init_place(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()


class TestSortDygraph(unittest.TestCase):

    def setUp(self):
        self.input_data = np.random.rand(10, 10)
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def func_api_0(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.sort(var_x)
        self.assertEqual((np.sort(self.input_data) == out.numpy()).all(), True)
        paddle.enable_static()

    def test_api_0(self):
        with _test_eager_guard():
            self.func_api_0()
        self.func_api_0()

    def func_api_1(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.sort(var_x, axis=-1)
        self.assertEqual((np.sort(self.input_data,
                                  axis=-1) == out.numpy()).all(), True)
        paddle.enable_static()

    def test_api_1(self):
        with _test_eager_guard():
            self.func_api_1()
        self.func_api_1()
