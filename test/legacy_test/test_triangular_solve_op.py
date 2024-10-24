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

sys.path.append("..")
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import Program, core, program_guard

paddle.enable_static()


# 2D + 2D , test 'upper'
class TestTriangularSolveOp(OpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = [12, 12]
        self.y_shape = [12, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        self.output = np.linalg.solve(
            np.triu(self.inputs['X']), self.inputs['Y']
        )

    def setUp(self):
        self.op_type = "triangular_solve"
        self.python_api = paddle.tensor.linalg.triangular_solve
        self.config()

        if self.dtype is np.complex64 or self.dtype is np.complex128:
            self.inputs = {
                'X': (
                    np.random.random(self.x_shape)
                    + 1j * np.random.random(self.x_shape)
                ).astype(self.dtype),
                'Y': (
                    np.random.random(self.y_shape)
                    + 1j * np.random.random(self.y_shape)
                ).astype(self.dtype),
            }
        else:
            self.inputs = {
                'X': np.random.random(self.x_shape).astype(self.dtype),
                'Y': np.random.random(self.y_shape).astype(self.dtype),
            }

        self.attrs = {
            'upper': self.upper,
            'transpose': self.transpose,
            'unitriangular': self.unitriangular,
        }
        self.set_output()
        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_cinn=True, check_pir=True)


# 2D(broadcast) + 3D, test 'transpose'
class TestTriangularSolveOp2(TestTriangularSolveOp):
    """
    case 2
    """

    def config(self):
        self.x_shape = [10, 10]
        self.y_shape = [3, 10, 8]
        self.upper = False
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X']).transpose(1, 0)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


# 3D(broadcast) + 3D
class TestTriangularSolveOp3(TestTriangularSolveOp):
    """
    case 3
    """

    def config(self):
        self.x_shape = [1, 10, 10]
        self.y_shape = [6, 10, 12]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


# 3D + 3D(broadcast), test 'transpose'
class TestTriangularSolveOp4(TestTriangularSolveOp):
    """
    case 4
    """

    def config(self):
        self.x_shape = [3, 10, 10]
        self.y_shape = [1, 10, 12]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.triu(self.inputs['X']).transpose(0, 2, 1)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


# 2D + 2D , test 'unitriangular' specially
class TestTriangularSolveOp5(TestTriangularSolveOp):
    """
    case 5
    """

    def config(self):
        self.x_shape = [10, 10]
        self.y_shape = [10, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = True
        self.dtype = "float64"

    def set_output(self):
        x = np.triu(self.inputs['X'])
        np.fill_diagonal(x, 1.0)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_grad_normal(self):
        x = np.triu(self.inputs['X'])
        np.fill_diagonal(x, 1.0)
        grad_out = np.ones([10, 10]).astype('float64')
        grad_y = np.linalg.solve(x.transpose(1, 0), grad_out)

        grad_x = -np.matmul(grad_y, self.output.transpose(1, 0))
        grad_x = np.triu(grad_x)
        np.fill_diagonal(grad_x, 0.0)

        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=[grad_x, grad_y],
            user_defined_grad_outputs=[grad_out],
        )


# 4D(broadcast) + 4D(broadcast)
class TestTriangularSolveOp6(TestTriangularSolveOp):
    """
    case 6
    """

    def config(self):
        self.x_shape = [1, 3, 10, 10]
        self.y_shape = [2, 1, 10, 5]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


# 3D(broadcast) + 4D(broadcast), test 'upper'
class TestTriangularSolveOp7(TestTriangularSolveOp):
    """
    case 7
    """

    def config(self):
        self.x_shape = [2, 10, 10]
        self.y_shape = [5, 1, 10, 2]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.triu(self.inputs['X']).transpose(0, 2, 1)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


# 3D(broadcast) + 5D
class TestTriangularSolveOp8(TestTriangularSolveOp):
    """
    case 8
    """

    def config(self):
        self.x_shape = [12, 3, 3]
        self.y_shape = [2, 3, 12, 3, 2]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


# 5D + 4D(broadcast)
class TestTriangularSolveOp9(TestTriangularSolveOp):
    """
    case 9
    """

    def config(self):
        self.x_shape = [2, 4, 2, 3, 3]
        self.y_shape = [4, 1, 3, 10]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.matmul(np.linalg.inv(x), y)


# 3D(broadcast) + 3D complex64
class TestTriangularSolveOpCp643b3(TestTriangularSolveOp):
    """
    case 10
    """

    def config(self):
        self.x_shape = [1, 10, 10]
        self.y_shape = [6, 10, 12]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex64

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 2D + 2D upper complex64
class TestTriangularSolveOpCp6422Up(TestTriangularSolveOp):
    """
    case 11
    """

    def config(self):
        self.x_shape = [12, 12]
        self.y_shape = [12, 10]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex64

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
            max_relative_error=0.02,
        )


# 2D(broadcast) + 3D, test 'transpose' complex64
class TestTriangularSolveOpCp6423T(TestTriangularSolveOp):
    """
    case 12
    """

    def config(self):
        self.x_shape = [10, 10]
        self.y_shape = [3, 10, 8]
        self.upper = False
        self.transpose = True
        self.unitriangular = False
        self.dtype = np.complex64

    def set_output(self):
        x = np.tril(self.inputs['X']).transpose(1, 0)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 2D + 2D , test 'unitriangular' complex64
class TestTriangularSolveOpCp6422Un(TestTriangularSolveOp):
    """
    case 13
    """

    def config(self):
        self.x_shape = [10, 10]
        self.y_shape = [10, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = True
        self.dtype = np.complex64

    def set_output(self):
        x = np.triu(self.inputs['X'])
        np.fill_diagonal(x, 1.0 + 0j)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


# 4D(broadcast) + 4D(broadcast) complex64
class TestTriangularSolveOpCp644b4b(TestTriangularSolveOp):
    """
    case 14
    """

    def config(self):
        self.x_shape = [1, 3, 10, 10]
        self.y_shape = [2, 3, 10, 5]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex64

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
            max_relative_error=0.008,
        )


# 3D(broadcast) + 4D(broadcast), test 'upper' complex64
class TestTriangularSolveOpCp643b4bUp(TestTriangularSolveOp):
    """
    case 15
    """

    def config(self):
        self.x_shape = [2, 10, 10]
        self.y_shape = [5, 1, 10, 2]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = np.complex64

    def set_output(self):
        x = np.triu(self.inputs['X']).transpose(0, 2, 1)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 3D(broadcast) + 5D complex64
class TestTriangularSolveOpCp643b5(TestTriangularSolveOp):
    """
    case 16
    """

    def config(self):
        self.x_shape = [12, 3, 3]
        self.y_shape = [2, 3, 12, 3, 2]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex64

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 5D + 4D(broadcast) complex64
class TestTriangularSolveOpCp6454b(TestTriangularSolveOp):
    """
    case 17
    """

    def config(self):
        self.x_shape = [2, 4, 2, 3, 3]
        self.y_shape = [4, 1, 3, 10]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex64

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.matmul(np.linalg.inv(x), y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 3D(broadcast) + 3D complex128
class TestTriangularSolveOpCp1283b3(TestTriangularSolveOp):
    """
    case 18
    """

    def config(self):
        self.x_shape = [1, 10, 10]
        self.y_shape = [6, 10, 12]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex128

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 2D + 2D upper complex128
class TestTriangularSolveOpCp12822Up(TestTriangularSolveOp):
    """
    case 19
    """

    def config(self):
        self.x_shape = [12, 12]
        self.y_shape = [12, 10]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex128

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 2D(broadcast) + 3D, test 'transpose' complex128
class TestTriangularSolveOpCp12823T(TestTriangularSolveOp):
    """
    case 20
    """

    def config(self):
        self.x_shape = [10, 10]
        self.y_shape = [3, 10, 8]
        self.upper = False
        self.transpose = True
        self.unitriangular = False
        self.dtype = np.complex128

    def set_output(self):
        x = np.tril(self.inputs['X']).transpose(1, 0)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 2D + 2D , test 'unitriangular' complex128
class TestTriangularSolveOpCp12822Un(TestTriangularSolveOp):
    """
    case 21
    """

    def config(self):
        self.x_shape = [10, 10]
        self.y_shape = [10, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = True
        self.dtype = np.complex128

    def set_output(self):
        x = np.triu(self.inputs['X'])
        np.fill_diagonal(x, 1.0 + 0j)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
        )


# 4D(broadcast) + 4D(broadcast) complex128
class TestTriangularSolveOpCp1284b4b(TestTriangularSolveOp):
    """
    case 22
    """

    def config(self):
        self.x_shape = [1, 3, 10, 10]
        self.y_shape = [2, 3, 10, 5]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex128

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 3D(broadcast) + 4D(broadcast), test 'upper' complex128
class TestTriangularSolveOpCp1283b4bUp(TestTriangularSolveOp):
    """
    case 23
    """

    def config(self):
        self.x_shape = [2, 10, 10]
        self.y_shape = [5, 1, 10, 2]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = np.complex128

    def set_output(self):
        x = np.triu(self.inputs['X']).transpose(0, 2, 1)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 3D(broadcast) + 5D complex128
class TestTriangularSolveOpCp1283b5(TestTriangularSolveOp):
    """
    case 24
    """

    def config(self):
        self.x_shape = [12, 3, 3]
        self.y_shape = [2, 3, 12, 3, 2]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex128

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


# 5D + 4D(broadcast) complex128
class TestTriangularSolveOpCp12854b(TestTriangularSolveOp):
    """
    case 25
    """

    def config(self):
        self.x_shape = [2, 4, 2, 3, 3]
        self.y_shape = [4, 1, 3, 10]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = np.complex128

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.matmul(np.linalg.inv(x), y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )


class TestTriangularSolveAPI(unittest.TestCase):
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
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name="x", shape=[3, 3], dtype=self.dtype)
            y = paddle.static.data(name="y", shape=[3, 2], dtype=self.dtype)
            z = paddle.linalg.triangular_solve(x, y)

            x_np = np.random.random([3, 3]).astype(self.dtype)
            y_np = np.random.random([3, 2]).astype(self.dtype)
            z_np = np.linalg.solve(np.triu(x_np), y_np)

            exe = base.Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"x": x_np, "y": y_np},
                fetch_list=[z],
            )
            np.testing.assert_allclose(fetches[0], z_np, rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_np = np.random.random([3, 3]).astype(self.dtype)
            y_np = np.random.random([3, 2]).astype(self.dtype)
            z_np = np.linalg.solve(np.tril(x_np), y_np)

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            z = paddle.linalg.triangular_solve(x, y, upper=False)

            np.testing.assert_allclose(z_np, z.numpy(), rtol=1e-05)
            self.assertEqual(z_np.shape, z.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestTriangularSolveOpError(unittest.TestCase):
    def test_errors1(self):
        with program_guard(Program(), Program()):
            # The input type of solve_op must be Variable.
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            y1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.linalg.triangular_solve, x1, y1)

    def test_errors2(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # The data type of input must be float32 or float64.
            x2 = paddle.static.data(name="x2", shape=[30, 30], dtype="bool")
            y2 = paddle.static.data(name="y2", shape=[30, 10], dtype="bool")
            self.assertRaises(TypeError, paddle.linalg.triangular_solve, x2, y2)

            x3 = paddle.static.data(name="x3", shape=[30, 30], dtype="int32")
            y3 = paddle.static.data(name="y3", shape=[30, 10], dtype="int32")
            self.assertRaises(TypeError, paddle.linalg.triangular_solve, x3, y3)

            x4 = paddle.static.data(name="x4", shape=[30, 30], dtype="float16")
            y4 = paddle.static.data(name="y4", shape=[30, 10], dtype="float16")
            self.assertRaises(TypeError, paddle.linalg.triangular_solve, x4, y4)

            # The number of dimensions of input'X must be >= 2.
            x5 = paddle.static.data(name="x5", shape=[30], dtype="float64")
            y5 = paddle.static.data(name="y5", shape=[30, 30], dtype="float64")
            self.assertRaises(
                ValueError, paddle.linalg.triangular_solve, x5, y5
            )

            # The number of dimensions of input'Y must be >= 2.
            x6 = paddle.static.data(name="x6", shape=[30, 30], dtype="float64")
            y6 = paddle.static.data(name="y6", shape=[30], dtype="float64")
            self.assertRaises(
                ValueError, paddle.linalg.triangular_solve, x6, y6
            )

            # The inner-most 2 dimensions of input'X should be equal to each other
            x7 = paddle.static.data(name="x7", shape=[2, 3, 4], dtype="float64")
            y7 = paddle.static.data(name="y7", shape=[2, 4, 3], dtype="float64")
            self.assertRaises(
                ValueError, paddle.linalg.triangular_solve, x7, y7
            )


if __name__ == "__main__":
    unittest.main()
