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

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
import sys

sys.path.append("..")
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.fluid.framework import _test_eager_guard


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
            'Y': np.random.random(self.input_y_matrix_shape).astype(self.dtype)
        }
        self.outputs = {
            'Out': np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        }

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_eager=True)


# x broadcast + 3D batch case
class TestSolveOpBatched_case0(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((11, 11)).astype(self.dtype),
            'Y': np.random.random((2, 11, 7)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        max_relative_error=1e-1,
                        check_eager=True)


# 3D batch + y vector case
class TestSolveOpBatched_case1(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((20, 6, 6)).astype(self.dtype),
            'Y': np.random.random((20, 6)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        max_relative_error=0.04,
                        check_eager=True)


# 3D batch + y broadcast case
class TestSolveOpBatched_case2(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 10, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 10)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        max_relative_error=0.02,
                        check_eager=True)


# x broadcast + 3D batch case
class TestSolveOpBatched_case3(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((1, 10, 10)).astype(self.dtype),
            'Y': np.random.random((2, 10, 10)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        max_relative_error=0.02,
                        check_eager=True)


# 3D normal batch case
class TestSolveOpBatched_case4(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((3, 6, 6)).astype(self.dtype),
            'Y': np.random.random((3, 6, 7)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_eager=True)


# 4D normal batch case
class TestSolveOpBatched_case5(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 2, 6, 6)).astype(self.dtype),
            'Y': np.random.random((2, 2, 6, 6)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_eager=True)


# 4D batch + y broadcast case
class TestSolveOpBatched_case6(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 2, 6, 6)).astype(self.dtype),
            'Y': np.random.random((1, 2, 6, 9)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_eager=True)


# 5D normal batch case
class TestSolveOpBatched_case7(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype),
            'Y': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        max_relative_error=0.04,
                        check_eager=True)


# 5D batch + y broadcast case
class TestSolveOpBatched_case8(OpTest):

    def setUp(self):
        self.python_api = paddle.linalg.solve
        self.op_type = "solve"
        self.dtype = "float64"
        np.random.seed(2021)
        self.inputs = {
            'X': np.random.random((2, 2, 2, 4, 4)).astype(self.dtype),
            'Y': np.random.random((1, 2, 2, 4, 7)).astype(self.dtype)
        }
        result = np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        max_relative_error=0.04,
                        check_eager=True)


class TestSolveOpError(unittest.TestCase):

    def func_errors(self):
        with program_guard(Program(), Program()):
            # The input type of solve_op must be Variable.
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.CPUPlace())
            y1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.CPUPlace())
            self.assertRaises(TypeError, paddle.linalg.solve, x1, y1)

            # The data type of input must be float32 or float64.
            x2 = fluid.data(name="x2", shape=[30, 30], dtype="bool")
            y2 = fluid.data(name="y2", shape=[30, 10], dtype="bool")
            self.assertRaises(TypeError, paddle.linalg.solve, x2, y2)

            x3 = fluid.data(name="x3", shape=[30, 30], dtype="int32")
            y3 = fluid.data(name="y3", shape=[30, 10], dtype="int32")
            self.assertRaises(TypeError, paddle.linalg.solve, x3, y3)

            x4 = fluid.data(name="x4", shape=[30, 30], dtype="int64")
            y4 = fluid.data(name="y4", shape=[30, 10], dtype="int64")
            self.assertRaises(TypeError, paddle.linalg.solve, x4, y4)

            x5 = fluid.data(name="x5", shape=[30, 30], dtype="float16")
            y5 = fluid.data(name="y5", shape=[30, 10], dtype="float16")
            self.assertRaises(TypeError, paddle.linalg.solve, x5, y5)

            # The number of dimensions of input'X must be >= 2.
            x6 = fluid.data(name="x6", shape=[30], dtype="float64")
            y6 = fluid.data(name="y6", shape=[30], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.solve, x6, y6)

            # The inner-most 2 dimensions of input'X should be equal to each other
            x7 = fluid.data(name="x7", shape=[2, 3, 4], dtype="float64")
            y7 = fluid.data(name="y7", shape=[2, 4, 3], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.solve, x7, y7)

    def test_dygraph(self):
        with _test_eager_guard():
            self.func_errors()
        self.func_errors()


# 2D + vector case, FP64
class TestSolveOpAPI_1(unittest.TestCase):

    def setUp(self):
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = "float64"
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            paddle_input_x = fluid.data(name="input_x",
                                        shape=[3, 3],
                                        dtype=self.dtype)
            paddle_input_y = fluid.data(name="input_y",
                                        shape=[3],
                                        dtype=self.dtype)
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)

            np_input_x = np.random.random([3, 3]).astype(self.dtype)
            np_input_y = np.random.random([3]).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={
                                  "input_x": np_input_x,
                                  "input_y": np_input_y
                              },
                              fetch_list=[paddle_result])
            np.testing.assert_allclose(fetches[0],
                                       np.linalg.solve(np_input_x, np_input_y),
                                       rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def func_dygraph(self):

        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([3, 3]).astype(self.dtype)
            input_y_np = np.random.random([3]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(numpy_output,
                                       paddle_output.numpy(),
                                       rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_dygraph(self):
        with _test_eager_guard():
            self.func_dygraph()
        self.func_dygraph()


# 2D normal case, FP64
class TestSolveOpAPI_2(unittest.TestCase):

    def setUp(self):
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = "float64"
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            paddle_input_x = fluid.data(name="input_x",
                                        shape=[10, 10],
                                        dtype=self.dtype)
            paddle_input_y = fluid.data(name="input_y",
                                        shape=[10, 4],
                                        dtype=self.dtype)
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)

            np_input_x = np.random.random([10, 10]).astype(self.dtype)
            np_input_y = np.random.random([10, 4]).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={
                                  "input_x": np_input_x,
                                  "input_y": np_input_y
                              },
                              fetch_list=[paddle_result])
            np.testing.assert_allclose(fetches[0],
                                       np.linalg.solve(np_input_x, np_input_y),
                                       rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def func_dygraph(self):

        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([10, 10]).astype(self.dtype)
            input_y_np = np.random.random([10, 4]).astype(self.dtype)
            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(numpy_output,
                                       paddle_output.numpy(),
                                       rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_dygraph(self):
        with _test_eager_guard():
            self.func_dygraph()
        self.func_dygraph()


# 2D normal case, FP32
class TestSolveOpAPI_3(unittest.TestCase):

    def setUp(self):
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = "float32"
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            paddle_input_x = fluid.data(name="input_x",
                                        shape=[10, 10],
                                        dtype=self.dtype)
            paddle_input_y = fluid.data(name="input_y",
                                        shape=[10, 4],
                                        dtype=self.dtype)
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)

            np_input_x = np.random.random([10, 10]).astype(self.dtype)
            np_input_y = np.random.random([10, 4]).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={
                                  "input_x": np_input_x,
                                  "input_y": np_input_y
                              },
                              fetch_list=[paddle_result])
            np.testing.assert_allclose(fetches[0],
                                       np.linalg.solve(np_input_x, np_input_y),
                                       rtol=0.0001)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def func_dygraph(self):

        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([10, 10]).astype(self.dtype)
            input_y_np = np.random.random([10, 4]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(numpy_output,
                                       paddle_output.numpy(),
                                       rtol=0.0001)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_dygraph(self):
        with _test_eager_guard():
            self.func_dygraph()
        self.func_dygraph()


# 3D + y broadcast case, FP64
class TestSolveOpAPI_4(unittest.TestCase):

    def setUp(self):
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = "float64"
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            paddle_input_x = fluid.data(name="input_x",
                                        shape=[2, 3, 3],
                                        dtype=self.dtype)
            paddle_input_y = fluid.data(name="input_y",
                                        shape=[1, 3, 3],
                                        dtype=self.dtype)
            paddle_result = paddle.linalg.solve(paddle_input_x, paddle_input_y)

            np_input_x = np.random.random([2, 3, 3]).astype(self.dtype)
            np_input_y = np.random.random([1, 3, 3]).astype(self.dtype)

            np_result = np.linalg.solve(np_input_x, np_input_y)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={
                                  "input_x": np_input_x,
                                  "input_y": np_input_y
                              },
                              fetch_list=[paddle_result])
            np.testing.assert_allclose(fetches[0],
                                       np.linalg.solve(np_input_x, np_input_y),
                                       rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def func_dygraph(self):

        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            input_x_np = np.random.random([2, 3, 3]).astype(self.dtype)
            input_y_np = np.random.random([1, 3, 3]).astype(self.dtype)

            tensor_input_x = paddle.to_tensor(input_x_np)
            tensor_input_y = paddle.to_tensor(input_y_np)

            numpy_output = np.linalg.solve(input_x_np, input_y_np)
            paddle_output = paddle.linalg.solve(tensor_input_x, tensor_input_y)
            np.testing.assert_allclose(numpy_output,
                                       paddle_output.numpy(),
                                       rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_dygraph(self):
        with _test_eager_guard():
            self.func_dygraph()
        self.func_dygraph()


class TestSolveOpSingularAPI(unittest.TestCase):
    # Singular matrix is ​​not invertible
    def setUp(self):
        self.places = [fluid.CPUPlace()]
        self.dtype = "float64"
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data(name="x", shape=[4, 4], dtype=self.dtype)
            y = fluid.data(name="y", shape=[4, 4], dtype=self.dtype)

            result = paddle.linalg.solve(x, y)

            input_x_np = np.ones([4, 4]).astype(self.dtype)
            input_y_np = np.ones([4, 4]).astype(self.dtype)

            exe = fluid.Executor(place)
            try:
                fetches = exe.run(fluid.default_main_program(),
                                  feed={
                                      "x": input_x_np,
                                      "y": input_y_np
                                  },
                                  fetch_list=[result])
            except RuntimeError as ex:
                print("The mat is singular")
                pass
            except ValueError as ex:
                print("The mat is singular")
                pass

    def test_static(self):
        for place in self.places:
            paddle.enable_static()
            self.check_static_result(place=place)

    def func_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input_x_np = np.ones([4, 4]).astype(self.dtype)
                input_y_np = np.ones([4, 4]).astype(self.dtype)
                input_x = fluid.dygraph.to_variable(input_x_np)
                input_y = fluid.dygraph.to_variable(input_y_np)
                try:
                    result = paddle.linalg.solve(input_x, input_y)
                except RuntimeError as ex:
                    print("The mat is singular")
                    pass
                except ValueError as ex:
                    print("The mat is singular")
                    pass

    def test_dygraph(self):
        with _test_eager_guard():
            self.func_dygraph()
        self.func_dygraph()


if __name__ == "__main__":
    unittest.main()
