# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestPaddleRavel(unittest.TestCase):
    def setUp(self):
        self.input_np = np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")
        self.input_shape = self.input_np.shape
        self.input_dtype = "float32"
        self.op_static = lambda x: paddle.ravel(x)
        self.op_dygraph = lambda x: paddle.ravel(x)
        self.expected = lambda x: x.flatten()
        self.places = [None, paddle.CPUPlace()]

    def check_static_result(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input_name = 'input'
            input_var = paddle.static.data(
                name=input_name, shape=self.input_shape, dtype=self.input_dtype
            )
            res = self.op_static(input_var)
            exe = base.Executor(place)
            fetches = exe.run(
                main_prog,
                feed={input_name: self.input_np},
                fetch_list=[res],
            )
            expect = (
                self.expected(self.input_np)
                if callable(self.expected)
                else self.expected
            )
            np.testing.assert_allclose(fetches[0], expect, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def check_dygraph_result(self, place):
        with base.dygraph.guard(place):
            input = paddle.to_tensor(self.input_np, stop_gradient=False)
            result = self.op_dygraph(input)
            expect = (
                self.expected(self.input_np)
                if callable(self.expected)
                else self.expected
            )
            # check forward
            np.testing.assert_allclose(result.numpy(), expect, rtol=1e-05)

            # check backward
            paddle.autograd.backward([result])
            np.testing.assert_allclose(
                input.grad.numpy(), np.ones_like(self.input_np), rtol=1e-05
            )

    def test_dygraph(self):
        for place in self.places:
            self.check_dygraph_result(place=place)


class TestPaddleRavel_case1(TestPaddleRavel):
    def setUp(self):
        # check Ravel 1d
        self.input_np = np.array([7, 8, 9], dtype="float32")
        self.input_shape = self.input_np.shape
        self.input_dtype = "float32"
        self.op_static = lambda x: paddle.ravel(x)
        self.op_dygraph = lambda x: paddle.ravel(x)
        self.expected = lambda x: x.flatten()
        self.places = [None, paddle.CPUPlace()]


class TestPaddleRavel_case2(TestPaddleRavel):
    def setUp(self):
        # check Ravel 3d
        self.input_np = np.arange(24, dtype="float32").reshape(2, 3, 4)
        self.input_shape = self.input_np.shape
        self.input_dtype = "float32"
        self.op_static = lambda x: paddle.ravel(x)
        self.op_dygraph = lambda x: paddle.ravel(x)
        self.expected = lambda x: x.flatten()
        self.places = [None, paddle.CPUPlace()]


class TestPaddleRavel_case3(TestPaddleRavel):
    def setUp(self):
        # check Ravel 0d (scalar)
        self.input_np = np.array(5.0, dtype="float32")
        self.input_shape = self.input_np.shape
        self.input_dtype = "float32"
        self.op_static = lambda x: paddle.ravel(x)
        self.op_dygraph = lambda x: paddle.ravel(x)
        self.expected = lambda x: x.flatten()
        self.places = [None, paddle.CPUPlace()]


class TestPaddleRavel_case4(TestPaddleRavel):
    def setUp(self):
        # check Ravel empty array
        self.input_np = np.array([], dtype="float32").reshape(0, 3)
        self.input_shape = self.input_np.shape
        self.input_dtype = "float32"
        self.op_static = lambda x: paddle.ravel(x)
        self.op_dygraph = lambda x: paddle.ravel(x)
        self.expected = lambda x: x.flatten()
        self.places = [None, paddle.CPUPlace()]


if __name__ == "__main__":
    unittest.main()
