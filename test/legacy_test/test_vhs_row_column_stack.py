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


class StackBase(unittest.TestCase):
    def setUp(self):
        self.x = np.array(1.0, dtype="float64")
        self.x_shape = []
        self.x_dtype = "float64"
        self.y = np.array(1.0, dtype="float64")
        self.y_shape = []
        self.y_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def static_api(self, func):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                name='x', shape=self.x_shape, dtype=self.x_dtype
            )
            y = paddle.static.data(
                name='y', shape=self.y_shape, dtype=self.y_dtype
            )
            f = getattr(paddle, func)
            out = f((x, y))
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[out],
            )
            f = getattr(np, func)
            expect_output = f((self.x, self.y))
            np.testing.assert_allclose(expect_output, res[0], atol=1e-05)

    def dygraph_api(self, func):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        f = getattr(paddle, func)
        res = f((x, y))
        f = getattr(np, func)
        expect_output = f((self.x, self.y))

        np.testing.assert_allclose(expect_output, res.numpy(), atol=1e-05)
        paddle.enable_static()


class TestColumnStackTwo(StackBase):
    def setUp(self):
        # 2 tensor stack
        self.x = np.array([1, 2, 3], dtype="int32")
        self.x_shape = [3]
        self.x_dtype = "int32"
        self.y = np.array([4, 5, 6], dtype="int32")
        self.y_shape = [3]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('column_stack')

    def test_dygraph_api(self):
        super().dygraph_api('column_stack')


class TestColumnStackThree(StackBase):
    def setUp(self):
        # three tensor stack
        self.x = np.array([0, 1, 2, 3, 4], dtype="int32")
        self.x_shape = [5]
        self.x_dtype = "int32"
        self.y = np.array(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype="int32"
        )
        self.y_shape = [5, 2]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('column_stack')

    def test_dygraph_api(self):
        super().dygraph_api('column_stack')


class TestHStack1D2T(StackBase):
    def setUp(self):
        # 2 tensor with 1 dim
        self.x = np.array([1, 2, 3], dtype="int32")
        self.x_shape = [3]
        self.x_dtype = "int32"
        self.y = np.array([4, 5, 6], dtype="int32")
        self.y_shape = [3]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('hstack')

    def test_dygraph_api(self):
        super().dygraph_api('hstack')


class TestHStack2D2T(StackBase):
    def setUp(self):
        # 2 tensor with 2 dim
        self.x = np.array([[1], [2], [3]], dtype="int32")
        self.x_shape = [3, 1]
        self.x_dtype = "int32"
        self.y = np.array([[4], [5], [6]], dtype="int32")
        self.y_shape = [3, 1]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('hstack')

    def test_dygraph_api(self):
        super().dygraph_api('hstack')


class TestDStack1D2T(StackBase):
    def setUp(self):
        # 2 tensor with 1 dim
        self.x = np.array([1, 2, 3], dtype="int32")
        self.x_shape = [3]
        self.x_dtype = "int32"
        self.y = np.array([4, 5, 6], dtype="int32")
        self.y_shape = [3]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('dstack')

    def test_dygraph_api(self):
        super().dygraph_api('dstack')


class TestDStack2D2T(StackBase):
    def setUp(self):
        # 2 tensor with 2 dim
        self.x = np.array([[1], [2], [3]], dtype="int32")
        self.x_shape = [3, 1]
        self.x_dtype = "int32"
        self.y = np.array([[4], [5], [6]], dtype="int32")
        self.y_shape = [3, 1]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('dstack')

    def test_dygraph_api(self):
        super().dygraph_api('dstack')


class TestRowStack1D2T(StackBase):
    def setUp(self):
        # 2 tensor with 1 dim
        self.x = np.array([1, 2, 3], dtype="int32")
        self.x_shape = [3]
        self.x_dtype = "int32"
        self.y = np.array([4, 5, 6], dtype="int32")
        self.y_shape = [3]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('row_stack')

    def test_dygraph_api(self):
        super().dygraph_api('row_stack')


class TestRowStack2D2T(StackBase):
    def setUp(self):
        # 2 tensor with 2 dim
        self.x = np.array([[1], [2], [3]], dtype="int32")
        self.x_shape = [3, 1]
        self.x_dtype = "int32"
        self.y = np.array([[4], [5], [6]], dtype="int32")
        self.y_shape = [3, 1]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('row_stack')

    def test_dygraph_api(self):
        super().dygraph_api('row_stack')


class TestVStack1D2T(StackBase):
    def setUp(self):
        # 2 tensor with 1 dim
        self.x = np.array([1, 2, 3], dtype="int32")
        self.x_shape = [3]
        self.x_dtype = "int32"
        self.y = np.array([4, 5, 6], dtype="int32")
        self.y_shape = [3]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('vstack')

    def test_dygraph_api(self):
        super().dygraph_api('vstack')


class TestVStack2D2T(StackBase):
    def setUp(self):
        # 2 tensor with 2 dim
        self.x = np.array([[1], [2], [3]], dtype="int32")
        self.x_shape = [3, 1]
        self.x_dtype = "int32"
        self.y = np.array([[4], [5], [6]], dtype="int32")
        self.y_shape = [3, 1]
        self.y_dtype = "int32"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('vstack')

    def test_dygraph_api(self):
        super().dygraph_api('vstack')


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
