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

from __future__ import print_function

import unittest
import numpy as np
import scipy
import scipy.linalg

import sys
sys.path.append("..")
import paddle
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard, core

paddle.enable_static()


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
        U, scipy.linalg.solve_triangular(
            L, B, lower=True))


def scipy_cholesky_solution(X, B, upper=True):
    if upper:
        umat = np.triu(X)
        A = umat.T @umat
    else:
        umat = np.tril(X)
        A = umat @umat.T
    K = scipy.linalg.cho_factor(A)
    return scipy.linalg.cho_solve(K, B)


def boardcast_shape(matA, matB):
    shapeA = matA.shape
    shapeB = matB.shape
    Boardshape = []
    for idx in range(len(shapeA) - 2):
        if shapeA[idx] == shapeB[idx]:
            Boardshape.append(shapeA[idx])
            continue
        elif shapeA[idx] == 1 or shapeB[idx] == 1:
            Boardshape.append(max(shapeA[idx], shapeB[idx]))
        else:
            raise Exception(
                'shapeA and shapeB should be boardcasted, but got {} and {}'.
                format(shapeA, shapeB))
    bsA = Boardshape + list(shapeA[-2:])
    bsB = Boardshape + list(shapeB[-2:])
    return np.broadcast_to(matA, bsA), np.broadcast_to(matB, bsB)


def scipy_cholesky_solution_batch(bumat, bB, upper=True):
    bumat, bB = boardcast_shape(bumat, bB)
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


# 2D + 2D , , upper=False
class TestCholeskySolveOp(OpTest):
    """
    case 1
    """

    def config(self):
        self.y_shape = [15, 15]
        self.x_shape = [15, 5]
        self.upper = False
        self.dtype = np.float64

    def set_output(self):
        umat = self.inputs['Y']
        self.output = scipy_cholesky_solution_batch(
            umat, self.inputs['X'], upper=self.upper)

    def setUp(self):
        self.op_type = "cholesky_solve"
        self.config()

        if self.upper:
            umat = np.triu(np.random.random(self.y_shape).astype(self.dtype))
        else:
            umat = np.tril(np.random.random(self.y_shape).astype(self.dtype))

        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': umat
        }
        self.attrs = {'upper': self.upper}
        self.set_output()
        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Y'], 'Out', max_relative_error=0.01)


# 3D(broadcast) + 3D, upper=True
class TestCholeskySolveOp3(TestCholeskySolveOp):
    """
    case 3
    """

    def config(self):
        self.y_shape = [1, 10, 10]
        self.x_shape = [2, 10, 5]
        self.upper = True
        self.dtype = np.float64


class TestCholeskySolveAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        # self.place = [paddle.CUDAPlace(0)]
        self.dtype = "float64"
        self.upper = True
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data(name="x", shape=[10, 2], dtype=self.dtype)
            y = fluid.data(name="y", shape=[10, 10], dtype=self.dtype)
            z = paddle.linalg.cholesky_solve(x, y, upper=self.upper)

            x_np = np.random.random([10, 2]).astype(self.dtype)
            y_np = np.random.random([10, 10]).astype(self.dtype)
            if self.upper:
                umat = np.triu(y_np)
            else:
                umat = np.tril(y_np)
            z_np = cholesky_solution(umat, x_np, upper=self.upper)
            z2_np = scipy_cholesky_solution(umat, x_np, upper=self.upper)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"x": x_np,
                                    "y": umat},
                              fetch_list=[z])
            self.assertTrue(np.allclose(fetches[0], z_np))

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_np = np.random.random([20, 2]).astype(self.dtype)
            y_np = np.random.random([20, 20]).astype(self.dtype)
            z_np = scipy_cholesky_solution(y_np, x_np, upper=self.upper)

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            z = paddle.linalg.cholesky_solve(x, y, upper=self.upper)

            self.assertTrue(np.allclose(z_np, z.numpy()))
            self.assertEqual(z_np.shape, z.numpy().shape)
            paddle.enable_static()

        for idx, place in enumerate(self.place):
            run(place)

    def test_boardcast(self):
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
            self.assertTrue(np.allclose(z_sci, z.numpy()))

        for idx, place in enumerate(self.place):
            run(place)


class TestCholeskySolveOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input type of solve_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            y1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, paddle.linalg.cholesky_solve, x1, y1)

            # The data type of input must be float32 or float64.        
            x2 = fluid.data(name="x2", shape=[30, 30], dtype="bool")
            y2 = fluid.data(name="y2", shape=[30, 10], dtype="bool")
            self.assertRaises(TypeError, paddle.linalg.cholesky_solve, x2, y2)

            x3 = fluid.data(name="x3", shape=[30, 30], dtype="int32")
            y3 = fluid.data(name="y3", shape=[30, 10], dtype="int32")
            self.assertRaises(TypeError, paddle.linalg.cholesky_solve, x3, y3)

            x4 = fluid.data(name="x4", shape=[30, 30], dtype="float16")
            y4 = fluid.data(name="y4", shape=[30, 10], dtype="float16")
            self.assertRaises(TypeError, paddle.linalg.cholesky_solve, x4, y4)

            # The number of dimensions of input'X must be >= 2.
            x5 = fluid.data(name="x5", shape=[30], dtype="float64")
            y5 = fluid.data(name="y5", shape=[30, 30], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.cholesky_solve, x5, y5)

            # The number of dimensions of input'Y must be >= 2.
            x6 = fluid.data(name="x6", shape=[30, 30], dtype="float64")
            y6 = fluid.data(name="y6", shape=[30], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.cholesky_solve, x6, y6)

            # The inner-most 2 dimensions of input'X should be equal to each other
            x7 = fluid.data(name="x7", shape=[2, 3, 4], dtype="float64")
            y7 = fluid.data(name="y7", shape=[2, 4, 3], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.cholesky_solve, x7, y7)


if __name__ == "__main__":
    unittest.main()
