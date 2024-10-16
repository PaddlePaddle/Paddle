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
from paddle import base
from paddle.base import core


class TestSortOnCPU(unittest.TestCase):
    def setUp(self):
        self.place = core.CPUPlace()

    def test_api_0(self):
        with base.program_guard(base.Program()):
            input = paddle.static.data(
                name="input", shape=[2, 3, 4], dtype="float32"
            )
            output = paddle.sort(x=input)
            exe = base.Executor(self.place)
            data = np.array(
                [
                    [[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
                    [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]],
                ],
                dtype='float32',
            )
            (result,) = exe.run(feed={'input': data}, fetch_list=[output])
            np_result = np.sort(result)
            self.assertEqual((result == np_result).all(), True)

    def test_api_1(self):
        with base.program_guard(base.Program()):
            input = paddle.static.data(
                name="input", shape=[2, 3, 4], dtype="float32"
            )
            output = paddle.sort(x=input, axis=1)
            exe = base.Executor(self.place)
            data = np.array(
                [
                    [[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
                    [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]],
                ],
                dtype='float32',
            )
            (result,) = exe.run(feed={'input': data}, fetch_list=[output])
            np_result = np.sort(result, axis=1)
            self.assertEqual((result == np_result).all(), True)

    def test_api_2(self):
        with base.program_guard(base.Program()):
            input = paddle.static.data(
                name="input", shape=[30], dtype="float32"
            )
            output = paddle.sort(x=input, axis=0, stable=True)
            exe = base.Executor(self.place)
            data = np.array(
                [100.0, 50.0, 10.0] * 10,
                dtype='float32',
            )
            (result,) = exe.run(feed={'input': data}, fetch_list=[output])
            np_result = np.sort(result, axis=0, kind='stable')
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

    def test_api_0(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.sort(var_x)
        self.assertEqual((np.sort(self.input_data) == out.numpy()).all(), True)
        paddle.enable_static()

    def test_api_1(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.sort(var_x, axis=-1)
        self.assertEqual(
            (np.sort(self.input_data, axis=-1) == out.numpy()).all(), True
        )
        paddle.enable_static()

    def test_api_2(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(np.array([100.0, 50.0, 10.0] * 10))
        out = paddle.sort(var_x, axis=0)
        self.assertEqual(
            (
                np.sort(
                    np.array([100.0, 50.0, 10.0] * 10), axis=0, kind='stable'
                )
                == out.numpy()
            ).all(),
            True,
        )
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
