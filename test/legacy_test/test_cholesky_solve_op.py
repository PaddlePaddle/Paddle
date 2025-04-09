#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np
import scipy
import scipy.linalg

sys.path.append("..")
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import Program, core, program_guard

paddle.enable_static()


# cholesky_solve implement 1
def cholesky_solution(X, B, upper=True):
    if upper:
        A = np.triu(X)
        L = A.T
        U = A
    else:
        A = np.tril(X)
        L = A
        U = A.T
    return scipy.linalg.solve_triangular(
        U, scipy.linalg.solve_triangular(L, B, lower=True)
    )


# cholesky_solve implement 2
def scipy_cholesky_solution(X, B, upper=True):
    if upper:
        umat = np.triu(X)
        A = umat.T @ umat
    else:
        umat = np.tril(X)
        A = umat @ umat.T
    K = scipy.linalg.cho_factor(A)
    return scipy.linalg.cho_solve(K, B)


# broadcast function used by cholesky_solve
def broadcast_shape(matA, matB):
    shapeA = matA.shape
    shapeB = matB.shape
    Broadshape = []
    for idx in range(len(shapeA) - 2):
        if shapeA[idx] == shapeB[idx]:
            Broadshape.append(shapeA[idx])
            continue
        elif shapeA[idx] == 1 or shapeB[idx] == 1:
            Broadshape.append(max(shapeA[idx], shapeB[idx]))
        else:
            raise Exception(
                f'shapeA and shapeB should be broadcasted, but got {shapeA} and {shapeB}'
            )
    bsA = Broadshape + list(shapeA[-2:])
    bsB = Broadshape + list(shapeB[-2:])
    return np.broadcast_to(matA, bsA), np.broadcast_to(matB, bsB)


# cholesky_solve implement in batch
def scipy_cholesky_solution_batch(bumat, bB, upper=True):
    bumat, bB = broadcast_shape(bumat, bB)
    ushape = bumat.shape
    bshape = bB.shape
    bumat = bumat.reshape((-1, ushape[-2], ushape[-1]))
    bB = bB.reshape((-1, bshape[-2], bshape[-1]))
    batch = 1
    for d in ushape[:-2]:
        batch *= d
    bx = []
    for b in range(batch):
        # x = scipy_cholesky_solution(bumat[b], bB[b], upper)   #large matrix result error
        x = cholesky_solution(bumat[b], bB[b], upper)
        bx.append(x)
    return np.array(bx).reshape(bshape)


# test condition: shape: 2D + 2D , upper=False
# based on OpTest class
class TestCholeskySolveOp(OpTest):
    """
    case 1
    """

    # test condition set
    def config(self):
        self.y_shape = [15, 15]
        self.x_shape = [15, 5]
        self.upper = False
        self.dtype = (
            np.float64
        )  # Here cholesky_solve Op only supports float64/float32 type, please check others if Op supports more types.

    # get scipy result
    def set_output(self):
        umat = self.inputs['Y']
        self.output = scipy_cholesky_solution_batch(
            umat, self.inputs['X'], upper=self.upper
        )

    def setUp(self):
        self.op_type = "cholesky_solve"
        self.python_api = paddle.tensor.cholesky_solve
        self.config()

        if self.upper:
            umat = np.triu(np.random.random(self.y_shape).astype(self.dtype))
        else:
            umat = np.tril(np.random.random(self.y_shape).astype(self.dtype))

        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': umat,
        }
        self.attrs = {'upper': self.upper}
        self.set_output()
        self.outputs = {'Out': self.output}

    # check Op forward result
    def test_check_output(self):
        self.check_output(check_pir=True)

    # check Op grad
    def test_check_grad_normal(self):
        self.check_grad(['Y'], 'Out', max_relative_error=0.01, check_pir=True)


# test condition:  3D(broadcast) + 3D, upper=True
class TestCholeskySolveOp3(TestCholeskySolveOp):
    """
    case 3
    """

    def config(self):
        self.y_shape = [1, 10, 10]
        self.x_shape = [2, 10, 5]
        self.upper = True
        self.dtype = np.float64


# API function test
class TestCholeskySolveAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = "float64"
        self.upper = True
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name="x", shape=[10, 2], dtype=self.dtype)
            y = paddle.static.data(name="y", shape=[10, 10], dtype=self.dtype)
            z = paddle.linalg.cholesky_solve(x, y, upper=self.upper)

            x_np = np.random.random([10, 2]).astype(self.dtype)
            y_np = np.random.random([10, 10]).astype(self.dtype)
            if self.upper:
                umat = np.triu(y_np)
            else:
                umat = np.tril(y_np)
            z_np = cholesky_solution(umat, x_np, upper=self.upper)
            z2_np = scipy_cholesky_solution(umat, x_np, upper=self.upper)

            exe = base.Executor(place)
            fetches = exe.run(
                feed={"x": x_np, "y": umat},
                fetch_list=[z],
            )
            np.testing.assert_allclose(fetches[0], z_np, rtol=1e-05)

    # test in static graph mode
    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    # test in dynamic mode
    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_np = np.random.random([20, 2]).astype(self.dtype)
            y_np = np.random.random([20, 20]).astype(self.dtype)
            z_np = scipy_cholesky_solution(y_np, x_np, upper=self.upper)

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            z = paddle.linalg.cholesky_solve(x, y, upper=self.upper)

            np.testing.assert_allclose(z_np, z.numpy(), rtol=1e-05)
            self.assertEqual(z_np.shape, z.numpy().shape)
            paddle.enable_static()

        for idx, place in enumerate(self.place):
            run(place)

    # test input with broadcast
    def test_broadcast(self):
        def run(place):
            paddle.disable_static()
            x_np = np.random.random([1, 30, 2]).astype(self.dtype)
            y_np = np.random.random([2, 30, 30]).astype(self.dtype)
            nx_np = np.concatenate((x_np, x_np), axis=0)

            z_sci = scipy_cholesky_solution_batch(y_np, nx_np, upper=self.upper)

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            z = paddle.linalg.cholesky_solve(x, y, upper=self.upper)
            self.assertEqual(z_sci.shape, z.numpy().shape)
            np.testing.assert_allclose(z_sci, z.numpy(), rtol=1e-05)

        for idx, place in enumerate(self.place):
            run(place)


# test condition out of bounds
class TestCholeskySolveOpError(unittest.TestCase):
    def test_errors_1(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input type of solve_op must be Variable.
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            y1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.linalg.cholesky_solve, x1, y1)

    def test_errors_2(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The data type of input must be float32 or float64.
            x2 = paddle.static.data(name="x2", shape=[30, 30], dtype="bool")
            y2 = paddle.static.data(name="y2", shape=[30, 10], dtype="bool")
            self.assertRaises(TypeError, paddle.linalg.cholesky_solve, x2, y2)

            x3 = paddle.static.data(name="x3", shape=[30, 30], dtype="int32")
            y3 = paddle.static.data(name="y3", shape=[30, 10], dtype="int32")
            self.assertRaises(TypeError, paddle.linalg.cholesky_solve, x3, y3)

            x4 = paddle.static.data(name="x4", shape=[30, 30], dtype="float16")
            y4 = paddle.static.data(name="y4", shape=[30, 10], dtype="float16")
            self.assertRaises(TypeError, paddle.linalg.cholesky_solve, x4, y4)

            # The number of dimensions of input'X must be >= 2.
            x5 = paddle.static.data(name="x5", shape=[30], dtype="float64")
            y5 = paddle.static.data(name="y5", shape=[30, 30], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.cholesky_solve, x5, y5)

            # The number of dimensions of input'Y must be >= 2.
            x6 = paddle.static.data(name="x6", shape=[30, 30], dtype="float64")
            y6 = paddle.static.data(name="y6", shape=[30], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.cholesky_solve, x6, y6)

            # The inner-most 2 dimensions of input'X should be equal to each other
            x7 = paddle.static.data(name="x7", shape=[2, 3, 4], dtype="float64")
            y7 = paddle.static.data(name="y7", shape=[2, 4, 3], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.cholesky_solve, x7, y7)


if __name__ == "__main__":
    unittest.main()
