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
# limitations under the License.w

import os
import sys
import unittest

import numpy as np

import paddle
from paddle.base import core

sys.path.append("..")
from op_test import OpTest

from paddle import base


# 2D normal case
class TestSolveOp(OpTest):
    def config(self):
        self.python_api = paddle.linalg.solve
        self.input_x_matrix_shape = [15, 15]
        self.input_y_matrix_shape = [15, 10]
        self.dtype = "float64"

    def setUp(self):
        paddle.enable_static()
        self.config()
        self.op_type = "solve"

        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random(self.input_x_matrix_shape).astype(self.dtype),
            'Y': np.random.random(self.input_y_matrix_shape).astype(self.dtype),
        }
        self.outputs = {
            'Out': np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


# x broadcast + 3D batch case
class TestSolveOpBatched_case0(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((11, 11)).astype(self.dtype),
            'Y': np.random.random((2, 11, 7)).astype(self.dtype),
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'], 'Out', max_relative_error=1e-1, check_pir=True
        )


# 3D batch + y vector case
class TestSolveOpBatched_case1(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((20, 6, 6)).astype(self.dtype),
            'Y': np.random.random((20, 6)).astype(self.dtype),
        }
        result = np.empty_like(self.inputs['Y'])
        for i in range(self.inputs['X'].shape[0]):
            result[i] = np.linalg.solve(
                self.inputs['X'][i], self.inputs['Y'][i]
            )
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'], 'Out', max_relative_error=0.04, check_pir=True
        )


# 3D batch + y broadcast case
class TestSolveOpBatched_case2(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 10, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 10)).astype(self.dtype),
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'], 'Out', max_relative_error=0.02, check_pir=True
        )


# x broadcast + 3D batch case
class TestSolveOpBatched_case3(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((1, 10, 10)).astype(self.dtype),
            'Y': np.random.random((2, 10, 10)).astype(self.dtype),
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'], 'Out', max_relative_error=0.02, check_pir=True
        )


# 3D normal batch case
class TestSolveOpBatched_case4(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((3, 6, 6)).astype(self.dtype),
            'Y': np.random.random((3, 6, 7)).astype(self.dtype),
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


# 4D normal batch case
class TestSolveOpBatched_case5(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 2, 6, 6)).astype(self.dtype),
            'Y': np.random.random((2, 2, 6, 6)).astype(self.dtype),
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


# 4D batch + y broadcast case
class TestSolveOpBatched_case6(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 2, 6, 6)).astype(self.dtype),
            'Y': np.random.random((1, 2, 6, 9)).astype(self.dtype),
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


# 5D normal batch case
class TestSolveOpBatched_case7(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype),
            'Y': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype),
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'], 'Out', max_relative_error=0.04, check_pir=True
        )


# 5D batch + y broadcast case
class TestSolveOpBatched_case8(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype),
            'Y': np.random.random((1, 2, 2, 4, 7)).astype(self.dtype),
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'], 'Out', max_relative_error=0.04, check_pir=True
        )


class TestSolveOpError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # The input type of solve_op must be Variable.
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            y1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.linalg.solve, x1, y1)

            # The data type of input must be float32 or float64.
            x2 = paddle.static.data(name="x2", shape=[30, 30], dtype="bool")
            y2 = paddle.static.data(name="y2", shape=[30, 10], dtype="bool")
            self.assertRaises(TypeError, paddle.linalg.solve, x2, y2)

            x3 = paddle.static.data(name="x3", shape=[30, 30], dtype="int32")
            y3 = paddle.static.data(name="y3", shape=[30, 10], dtype="int32")
            self.assertRaises(TypeError, paddle.linalg.solve, x3, y3)

            x4 = paddle.static.data(name="x4", shape=[30, 30], dtype="int64")
            y4 = paddle.static.data(name="y4", shape=[30, 10], dtype="int64")
            self.assertRaises(TypeError, paddle.linalg.solve, x4, y4)

            x5 = paddle.static.data(name="x5", shape=[30, 30], dtype="float16")
            y5 = paddle.static.data(name="y5", shape=[30, 10], dtype="float16")
            self.assertRaises(TypeError, paddle.linalg.solve, x5, y5)

            # The number of dimensions of input'X must be >= 2.
            x6 = paddle.static.data(name="x6", shape=[30], dtype="float64")
            y6 = paddle.static.data(name="y6", shape=[30], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.solve, x6, y6)

            # The inner-most 2 dimensions of input'X should be equal to each other
            x7 = paddle.static.data(name="x7", shape=[2, 3, 4], dtype="float64")
            y7 = paddle.static.data(name="y7", shape=[2, 4, 3], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.solve, x7, y7)

            # The shape of y should not be 1 when left = False. (if y is vector it should be a row vector)
            x8 = paddle.static.data(name="x8", shape=[3, 3], dtype="float64")
            y8 = paddle.static.data(name="y8", shape=[3], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.solve, x8, y8, False)

            # The height of x should equal the width of y when left = False.
            x9 = paddle.static.data(name="x9", shape=[2, 5, 5], dtype="float64")
            y9 = paddle.static.data(name="y9", shape=[5, 3], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.solve, x9, y9, False)


# 2D + vector case, FP64
class TestSolveOpAPI_1(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float64"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=[3, 3], dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=[3], dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)

            np_input_x = np.random.random([3, 3]).astype(self.dtype)
            np_input_y = np.random.random([3]).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(
                fetches[0], np.linalg.solve(np_input_x, np_input_y), rtol=1e-05
            )

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([3, 3]).astype(self.dtype)
            input_y_np = np.random.random([3]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


# 2D normal case, FP64
class TestSolveOpAPI_2(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float64"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=[10, 10], dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=[10, 4], dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)

            np_input_x = np.random.random([10, 10]).astype(self.dtype)
            np_input_y = np.random.random([10, 4]).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(
                fetches[0], np.linalg.solve(np_input_x, np_input_y), rtol=1e-05
            )

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([10, 10]).astype(self.dtype)
            input_y_np = np.random.random([10, 4]).astype(self.dtype)
            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


# 2D normal case, FP32
class TestSolveOpAPI_3(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float32"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=[10, 10], dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=[10, 4], dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)

            np_input_x = np.random.random([10, 10]).astype(self.dtype)
            np_input_y = np.random.random([10, 4]).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(
                fetches[0], np.linalg.solve(np_input_x, np_input_y), rtol=0.0001
            )

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([10, 10]).astype(self.dtype)
            input_y_np = np.random.random([10, 4]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=0.0001
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


# 3D + y broadcast case, FP64
class TestSolveOpAPI_4(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float64"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=[2, 3, 3], dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=[1, 3, 3], dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)

            np_input_x = np.random.random([2, 3, 3]).astype(self.dtype)
            np_input_y = np.random.random([1, 3, 3]).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(
                fetches[0], np.linalg.solve(np_input_x, np_input_y), rtol=1e-05
            )

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([2, 3, 3]).astype(self.dtype)
            input_y_np = np.random.random([1, 3, 3]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


def np_transpose_last_2dim(x):
    x_new_dims = list(range(len(x.shape)))
    x_new_dims[-1], x_new_dims[-2] = x_new_dims[-2], x_new_dims[-1]
    x = np.transpose(x, x_new_dims)
    return x


def np_solve_right(x, y):
    x = np_transpose_last_2dim(x)
    y = np_transpose_last_2dim(y)
    out = np.linalg.solve(x, y)
    out = np_transpose_last_2dim(out)
    return out


# 2D + vector right case, FP64
class TestSolveOpAPIRight_1(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float64"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=[3, 3], dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=[1, 3], dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(
                paddle_input_x, paddle_input_y, left=False
            )

            np_input_x = np.random.random([3, 3]).astype(self.dtype)
            np_input_y = np.random.random([1, 3]).astype(self.dtype)

            np_result = np_solve_right(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(fetches[0], np_result, rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([3, 3]).astype(self.dtype)
            input_y_np = np.random.random([1, 3]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np_solve_right(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(
                tensor_input_x, tensor_input_y, left=False
            )
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


# 2D normal right case, FP64
class TestSolveOpAPIRight_2(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float64"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=[10, 10], dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=[4, 10], dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(
                paddle_input_x, paddle_input_y, left=False
            )

            np_input_x = np.random.random([10, 10]).astype(self.dtype)
            np_input_y = np.random.random([4, 10]).astype(self.dtype)

            np_result = np_solve_right(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(fetches[0], np_result, rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([10, 10]).astype(self.dtype)
            input_y_np = np.random.random([4, 10]).astype(self.dtype)
            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np_solve_right(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(
                tensor_input_x, tensor_input_y, left=False
            )
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


# 2D normal right case, FP32
class TestSolveOpAPIRight_3(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float32"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=[10, 10], dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=[6, 10], dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(
                paddle_input_x, paddle_input_y, left=False
            )

            np_input_x = np.random.random([10, 10]).astype(self.dtype)
            np_input_y = np.random.random([6, 10]).astype(self.dtype)

            np_result = np_solve_right(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(fetches[0], np_result, rtol=0.0001)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([10, 10]).astype(self.dtype)
            input_y_np = np.random.random([6, 10]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np_solve_right(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(
                tensor_input_x, tensor_input_y, left=False
            )
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=0.0001
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


# 3D + y broadcast right case, FP64
class TestSolveOpAPIRight_4(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float64"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=[2, 3, 3], dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=[1, 3, 3], dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(
                paddle_input_x, paddle_input_y, left=False
            )

            np_input_x = np.random.random([2, 3, 3]).astype(self.dtype)
            np_input_y = np.random.random([1, 3, 3]).astype(self.dtype)

            np_result = np_solve_right(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(fetches[0], np_result, rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([2, 3, 3]).astype(self.dtype)
            input_y_np = np.random.random([1, 3, 3]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np_solve_right(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(
                tensor_input_x, tensor_input_y, left=False
            )
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestSolveOpSingularAPI(unittest.TestCase):
    # Singular matrix is ​​not invertible
    def setUp(self):
        self.places = []
        self.dtype = "float64"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.static.data(name="x", shape=[4, 4], dtype=self.dtype)
            y = paddle.static.data(name="y", shape=[4, 4], dtype=self.dtype)

            result = paddle.linalg.solve(x, y)

            input_x_np = np.ones([4, 4]).astype(self.dtype)
            input_y_np = np.ones([4, 4]).astype(self.dtype)

            exe = base.Executor(place)
            try:
                fetches = exe.run(
                    base.default_main_program(),
                    feed={"x": input_x_np, "y": input_y_np},
                    fetch_list=[result],
                )
            except RuntimeError as ex:
                print("The mat is singular")
            except ValueError as ex:
                print("The mat is singular")

    def test_static(self):
        for place in self.places:
            paddle.enable_static()
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_x_np = np.ones([4, 4]).astype(self.dtype)
                input_y_np = np.ones([4, 4]).astype(self.dtype)
                input_x = paddle.to_tensor(input_x_np)
                input_y = paddle.to_tensor(input_y_np)
                try:
                    result = paddle.linalg.solve(input_x, input_y)
                except RuntimeError as ex:
                    print("The mat is singular")
                except ValueError as ex:
                    print("The mat is singular")


class TestSolveOpAPIZeroDimCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = []
        self.dtype = "float32"
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place, x_shape, y_shape, np_y_shape):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            paddle_input_x = paddle.static.data(
                name="input_x", shape=x_shape, dtype=self.dtype
            )
            paddle_input_y = paddle.static.data(
                name="input_y", shape=y_shape, dtype=self.dtype
            )
            paddle_result = paddle.linalg.solve(
                paddle_input_x, paddle_input_y, left=False
            )

            np_input_x = np.random.random(x_shape).astype(self.dtype)
            np_input_y = np.random.random(np_y_shape).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": np_input_x, "input_y": np_input_y},
                fetch_list=[paddle_result],
            )
            np.testing.assert_allclose(fetches[0], np_result, rtol=0.0001)

    def test_static(self):
        for place in self.place:
            self.check_static_result(
                place=place,
                x_shape=[10, 0, 0],
                y_shape=[6, 0, 0],
                np_y_shape=[10, 0, 0],
            )
            with self.assertRaises(ValueError) as context:
                self.check_static_result(
                    place=place,
                    x_shape=[10, 0, 0],
                    y_shape=[10],
                    np_y_shape=[10],
                )

    def test_dygraph(self):
        def run(place, x_shape, y_shape):
            with base.dygraph.guard(place):
                input_x_np = np.random.random(x_shape).astype(self.dtype)
                input_y_np = np.random.random(y_shape).astype(self.dtype)

                tensor_input_x = paddle.to_tensor(input_x_np)
                tensor_input_y = paddle.to_tensor(input_y_np)

                numpy_output = np.linalg.solve(input_x_np, input_y_np)
                paddle_output = paddle.linalg.solve(
                    tensor_input_x, tensor_input_y, left=False
                )
                np.testing.assert_allclose(
                    numpy_output, paddle_output.numpy(), rtol=0.0001
                )
                self.assertEqual(
                    numpy_output.shape, paddle_output.numpy().shape
                )

        for place in self.place:
            run(place, x_shape=[10, 0, 0], y_shape=[10, 0, 0])
            with self.assertRaises(ValueError) as context:
                run(place, x_shape=[10, 0, 0], y_shape=[10])


if __name__ == "__main__":
    unittest.main()
