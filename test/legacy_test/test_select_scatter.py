# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestSelectScatterAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.x_shape = (10, 10)
        self.x = np.arange(100).reshape(self.x_shape).astype(self.dtype)
        self.value_shape = (10,)
        self.value = np.ones(self.value_shape).astype(self.dtype)
        self.dim = 0
        self.index = 0
        self.out = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
                [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0],
                [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0],
                [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0],
                [70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0],
                [80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0],
                [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0],
            ]
        )

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(
                name='x', shape=self.x_shape, dtype=self.dtype
            )
            value = paddle.static.data(
                name='value', shape=self.value_shape, dtype=self.dtype
            )
            out = paddle.select_scatter(
                x, value, dim=self.dim, index=self.index
            )

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)
            res = exe.run(
                base.default_main_program(),
                feed={'x': self.x, 'value': self.value},
                fetch_list=[out],
            )
            np.testing.assert_allclose(res[0], self.out, atol=1e-5, rtol=1e-5)
            paddle.disable_static()

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        value = paddle.to_tensor(self.value)
        result = paddle.select_scatter(x, value, dim=self.dim, index=self.index)
        np.testing.assert_allclose(self.out, result.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_error(self):
        x = paddle.to_tensor(self.x)
        value = paddle.to_tensor(self.value)
        self.assertRaises(ValueError, paddle.select_scatter, x, value, 0, 11)


class TestSelectScatterAPI1(TestSelectScatterAPI):
    def setUp(self):
        self.dtype = "float32"
        self.x_shape = (3, 4, 5)
        self.x = np.arange(60).reshape(self.x_shape).astype(self.dtype)
        self.value_shape = (4, 5)
        self.value = np.ones(self.value_shape).astype(self.dtype)
        self.dim = 0
        self.index = 1
        self.out = np.array(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [40.0, 41.0, 42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0, 48.0, 49.0],
                    [50.0, 51.0, 52.0, 53.0, 54.0],
                    [55.0, 56.0, 57.0, 58.0, 59.0],
                ],
            ]
        )


class TestSelectScatterAPI2(TestSelectScatterAPI):
    def setUp(self):
        self.dtype = "float32"
        self.x_shape = (3, 4, 5)
        self.x = np.arange(60).reshape(self.x_shape).astype(self.dtype)
        self.value_shape = (3, 5)
        self.value = np.ones(self.value_shape).astype(self.dtype)
        self.dim = 1
        self.index = 2
        self.out = np.array(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                ],
                [
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [35.0, 36.0, 37.0, 38.0, 39.0],
                ],
                [
                    [40.0, 41.0, 42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0, 48.0, 49.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [55.0, 56.0, 57.0, 58.0, 59.0],
                ],
            ]
        )


class TestSelectScatterAPI3(TestSelectScatterAPI):
    def setUp(self):
        self.dtype = "float32"
        self.x_shape = (3, 4, 5)
        self.x = np.arange(60).reshape(self.x_shape).astype(self.dtype)
        self.value_shape = (3, 4)
        self.value = np.ones(self.value_shape).astype(self.dtype)
        self.dim = 2
        self.index = 3
        self.out = np.array(
            [
                [
                    [0.0, 1.0, 2.0, 1.0, 4.0],
                    [5.0, 6.0, 7.0, 1.0, 9.0],
                    [10.0, 11.0, 12.0, 1.0, 14.0],
                    [15.0, 16.0, 17.0, 1.0, 19.0],
                ],
                [
                    [20.0, 21.0, 22.0, 1.0, 24.0],
                    [25.0, 26.0, 27.0, 1.0, 29.0],
                    [30.0, 31.0, 32.0, 1.0, 34.0],
                    [35.0, 36.0, 37.0, 1.0, 39.0],
                ],
                [
                    [40.0, 41.0, 42.0, 1.0, 44.0],
                    [45.0, 46.0, 47.0, 1.0, 49.0],
                    [50.0, 51.0, 52.0, 1.0, 54.0],
                    [55.0, 56.0, 57.0, 1.0, 59.0],
                ],
            ]
        )


if __name__ == "__main__":
    unittest.main()
