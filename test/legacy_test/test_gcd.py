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
from paddle.base import core


class TestGcdAPI(unittest.TestCase):
    def setUp(self):
        self.x_np = [12]
        self.y_np = [20]
        self.x_shape = [1]
        self.y_shape = [1]

    def test_static_graph(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(
                name='input1', dtype='int32', shape=self.x_shape
            )
            y = paddle.static.data(
                name='input2', dtype='int32', shape=self.y_shape
            )
            out = paddle.gcd(x, y)
            out_ref = np.gcd(self.x_np, self.y_np)

            exe = paddle.static.Executor(place)
            res = exe.run(
                paddle.static.default_main_program(),
                feed={'input1': self.x_np, 'input2': self.y_np},
                fetch_list=[out],
            )
            self.assertTrue((res[0] == out_ref).all())

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        result = paddle.gcd(x, y)
        np.testing.assert_allclose(
            np.gcd(self.x_np, self.y_np), result.numpy(), rtol=1e-05
        )

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
        self.x_shape = []
        self.y_shape = []


class TestGcdAPI4(TestGcdAPI):
    def setUp(self):
        self.x_np = [0]
        self.y_np = [0]
        self.x_shape = [1]
        self.y_shape = [1]


class TestGcdAPI5(TestGcdAPI):
    def setUp(self):
        self.x_np = 12
        self.y_np = -20
        self.x_shape = []
        self.y_shape = []


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
