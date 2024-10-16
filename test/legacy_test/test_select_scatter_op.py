#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import os
import unittest

import numpy as np

import paddle
from paddle.framework import core

paddle.enable_static()


class TestSelectScatterAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 3, 4]
        self.type = np.float32
        self.x_np = np.random.random(self.shape).astype(self.type)
        self.place = []
        self.axis = 1
        self.index = 1
        self.value_shape = [2, 4]
        self.value_np = np.random.random(self.value_shape).astype(self.type)
        self.x_feed = copy.deepcopy(self.x_np)
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def get_out_ref(self, out_ref, index, value_np):
        for i in range(2):
            for j in range(4):
                out_ref[i, index, j] = value_np[i, j]

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('Src', self.shape, self.type)
                value = paddle.static.data(
                    'Values', self.value_shape, self.type
                )
                out = paddle.select_scatter(x, value, self.axis, self.index)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={
                        'Src': self.x_feed,
                        'Values': self.value_np,
                    },
                    fetch_list=[out],
                )

            out_ref = copy.deepcopy(self.x_np)
            self.get_out_ref(out_ref, self.index, self.value_np)
            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.select_scatter(
                x_tensor, value_tensor, self.axis, self.index
            )
            out_ref = copy.deepcopy(self.x_np)
            self.get_out_ref(out_ref, self.index, self.value_np)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)


class TestSelectScatterAPICase2(TestSelectScatterAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 3, 4, 5]
        self.type = np.float64
        self.x_np = np.random.random(self.shape).astype(self.type)
        self.place = []
        self.axis = 2
        self.index = 1
        self.value_shape = [2, 3, 5]
        self.value_np = np.random.random(self.value_shape).astype(self.type)
        self.x_feed = copy.deepcopy(self.x_np)
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def get_out_ref(self, out_ref, index, value_np):
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    out_ref[i, j, index, k] = value_np[i, j, k]


class TestSelectScatterAPICase3(TestSelectScatterAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 3, 4, 5, 6]
        self.type = np.int32
        self.x_np = np.random.random(self.shape).astype(self.type)
        self.place = []
        self.axis = 2
        self.index = 1
        self.value_shape = [2, 3, 5, 6]
        self.value_np = np.random.random(self.value_shape).astype(self.type)
        self.x_feed = copy.deepcopy(self.x_np)
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def get_out_ref(self, out_ref, index, value_np):
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    for w in range(6):
                        out_ref[i, j, index, k, w] = value_np[i, j, k, w]


class TestSelectScatterAPIError(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 3, 4]
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = []
        self.axis = 1
        self.index = 1
        self.value_shape = [2, 4]
        self.value_np = np.random.random(self.value_shape).astype(np.float32)
        self.x_feed = copy.deepcopy(self.x_np)
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_len_of_shape_not_equal_error(self):
        with self.assertRaises(RuntimeError):
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor(self.value_np).reshape((2, 2, 2))
            res = paddle.select_scatter(x_tensor, value_tensor, 1, 1)

    def test_one_of_size_not_equal_error(self):
        with self.assertRaises(RuntimeError):
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor([[2, 2], [2, 2]]).astype(np.float32)
            res = paddle.select_scatter(x_tensor, value_tensor, 1, 1)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
