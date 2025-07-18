#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

import paddle
from paddle import base

sys.path.append("..")
from op_test import OpTest, get_places


def _transpose_last_2dim(x):
    """transpose the last 2 dimension of a tensor"""
    x_new_dims = list(range(len(x.shape)))
    x_new_dims[-1], x_new_dims[-2] = x_new_dims[-2], x_new_dims[-1]
    x = paddle.transpose(x, x_new_dims)
    return x


def get_inandout(A_shape, b_shape, trans="N", dtype="float64"):
    paddle.disable_static(base.CPUPlace())
    np.random.seed(2025)
    A = np.random.random(A_shape).astype(dtype)
    b = np.random.random(b_shape).astype(dtype)
    x_grad = paddle.to_tensor(np.random.random(b_shape).astype(dtype))
    paddle_A = paddle.to_tensor(A)
    lu, pivots = paddle.linalg.lu(paddle_A)
    if trans == "N":  # Ax = b
        out = np.linalg.solve(A, b)

        temp_A = np.swapaxes(A, -2, -1)
        b_grad = np.linalg.solve(temp_A, x_grad)

        _, L, U = paddle.linalg.lu_unpack(lu, pivots, True, False)
        U_mH = _transpose_last_2dim(paddle.conj(U))
        gR = paddle.linalg.triangular_solve(
            U_mH,
            paddle.mm(
                -x_grad,
                _transpose_last_2dim(paddle.conj(paddle.to_tensor(out))),
            ),
            False,
            False,
            False,
        )
        gL = paddle.linalg.triangular_solve(
            _transpose_last_2dim(paddle.conj(L)),
            paddle.mm(gR, U_mH),
            True,
            False,
            True,
        )
        lu_grad = (paddle.tril(gL, -1) + paddle.triu(gR, 0)).numpy()
    elif trans == "T":  # A^Tx = b
        temp_A = np.swapaxes(A, -2, -1)
        out = np.linalg.solve(temp_A, b)

        b_grad = np.linalg.solve(A, x_grad)

        P, L, U = paddle.linalg.lu_unpack(lu, pivots, True, True)
        gR = paddle.mm(-_transpose_last_2dim(P), paddle.to_tensor(out))
        gR = paddle.mm(gR, _transpose_last_2dim(paddle.conj(x_grad)))
        gR = paddle.mm(gR, P)
        L_mH = _transpose_last_2dim(paddle.conj(L))
        gR = paddle.linalg.triangular_solve(L_mH, gR, True, True, True)
        gU = paddle.linalg.triangular_solve(
            _transpose_last_2dim(paddle.conj(U)),
            paddle.mm(L_mH, gR),
            False,
            True,
            False,
        )
        lu_grad = (paddle.tril(gR, -1) + paddle.triu(gU, 0)).numpy()

    lu = lu.numpy()
    pivots = pivots.numpy()
    x_grad = x_grad.numpy()
    paddle.enable_static()
    return lu, pivots, b, out, x_grad, b_grad, lu_grad


class TestLuSolveOp(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.lu_solve
        self.op_type = "lu_solve"
        self.init_value()
        (
            self.LU,
            self.pivots,
            self.b,
            self.out,
            self.x_grad,
            self.b_grad,
            self.lu_grad,
        ) = get_inandout(self.A_shape, self.b_shape, self.trans, self.dtype)
        self.inputs = {
            'b': self.b,
            'lu': self.LU,
            'pivots': self.pivots,
        }
        self.attrs = {'trans': self.trans}
        self.outputs = {'out': self.out}

    def init_value(self):
        self.A_shape = [2, 10, 10]
        self.b_shape = [2, 10, 5]
        self.trans = "N"
        self.dtype = "float64"

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)
        paddle.disable_static()

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(
            ['b', 'lu'],
            'out',
            no_grad_set=['pivots'],
            user_defined_grads=[self.b_grad, self.lu_grad],
            user_defined_grad_outputs=[self.x_grad],
            check_pir=True,
        )
        paddle.disable_static()


class TestLuSolveOp1(TestLuSolveOp):
    def init_value(self):
        self.A_shape = [2, 10, 10]
        self.b_shape = [2, 10, 5]
        self.trans = "T"
        self.dtype = "float64"


class TestLuSolveOp2(TestLuSolveOp):
    def init_value(self):
        self.A_shape = [2, 2, 10, 10]
        self.b_shape = [2, 2, 10, 5]
        self.trans = "T"
        self.dtype = "float64"


class TestLuSolveOp3(TestLuSolveOp):
    def init_value(self):
        self.A_shape = [2, 2, 10, 10]
        self.b_shape = [2, 2, 10, 5]
        self.trans = "N"
        self.dtype = "float64"


class TestLuSolveOp4(TestLuSolveOp):
    def init_value(self):
        self.A_shape = [10, 10]
        self.b_shape = [10, 10]
        self.trans = "T"
        self.dtype = "float64"


class TestLuSolveOp5(TestLuSolveOp):
    def init_value(self):
        self.A_shape = [10, 10]
        self.b_shape = [10, 10]
        self.trans = "N"
        self.dtype = "float64"


class TestLuSolveOpAPI(unittest.TestCase):
    def setUp(self):
        self.init_value()
        (
            self.LU,
            self.pivots,
            self.b,
            self.out,
            _,
            _,
            _,
        ) = get_inandout(self.A_shape, self.b_shape, self.trans, self.dtype)
        self.place = get_places()

    def init_value(self):
        # Ax = b
        self.A_shape = [10, 10]
        self.b_shape = [10, 5]
        self.trans = "N"
        self.dtype = "float64"
        self.rtol = 1e-05

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            lu = paddle.to_tensor(self.LU)
            pivots = paddle.to_tensor(self.pivots)
            b = paddle.to_tensor(self.b)
            lu_solve_x = paddle.linalg.lu_solve(b, lu, pivots, self.trans)
            np.testing.assert_allclose(
                lu_solve_x.numpy(), self.out, rtol=self.rtol
            )
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_static(self):
        def run(place):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                b = paddle.static.data(
                    name='B', shape=self.b.shape, dtype=self.b.dtype
                )
                lu = paddle.static.data(
                    name='Lu', shape=self.LU.shape, dtype=self.LU.dtype
                )
                pivots = paddle.static.data(
                    name='Pivots',
                    shape=self.pivots.shape,
                    dtype=self.pivots.dtype,
                )
                lu_solve_x = paddle.linalg.lu_solve(b, lu, pivots, self.trans)
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={
                        'B': self.b,
                        'Lu': self.LU,
                        'Pivots': self.pivots,
                    },
                    fetch_list=[lu_solve_x],
                )
                np.testing.assert_allclose(fetches[0], self.out, rtol=self.rtol)
            paddle.disable_static()

        for place in self.place:
            run(place)


class TestLuSolveOpAPI2(TestLuSolveOpAPI):
    def init_value(self):
        # Ax = b
        self.A_shape = [1, 10, 10]
        self.b_shape = [2, 10, 5]
        self.trans = "N"
        self.dtype = "float64"
        self.rtol = 1e-05


class TestLuSolveOpAPI3(TestLuSolveOpAPI):
    def init_value(self):
        # A^Tx = b
        self.A_shape = [1, 10, 10]
        self.b_shape = [2, 10, 5]
        self.trans = "T"
        self.dtype = "float64"
        self.rtol = 1e-05


class TestLuSolveOpAPI4(TestLuSolveOpAPI):
    def init_value(self):
        # Ax = b
        self.A_shape = [1, 10, 10]
        self.b_shape = [2, 10, 5]
        self.trans = "N"
        self.dtype = "float32"
        self.rtol = 0.001


class TestLuSolveOpAPI5(TestLuSolveOpAPI):
    def init_value(self):
        # A^Tx = b
        self.A_shape = [1, 10, 10]
        self.b_shape = [2, 10, 5]
        self.trans = "T"
        self.dtype = "float32"
        self.rtol = 0.001


class TestLuSolveOpAPI6(TestLuSolveOpAPI):
    def init_value(self):
        # Ax = b
        self.A_shape = [10, 10]
        self.b_shape = [10, 5]
        self.trans = "N"
        self.dtype = "float32"
        self.rtol = 0.001


class TestLuSolveOpAPI7(TestLuSolveOpAPI):
    def init_value(self):
        # A^Tx = b
        self.A_shape = [10, 10]
        self.b_shape = [10, 5]
        self.trans = "T"
        self.dtype = "float32"
        self.rtol = 0.001


class TestLuSolveOpAPI8(TestLuSolveOpAPI):
    def init_value(self):
        # A^Tx = b
        self.A_shape = [10, 10]
        self.b_shape = [10, 5]
        self.trans = "T"
        self.dtype = "float64"
        self.rtol = 1e-05


class TestLSolveError(unittest.TestCase):
    def test_errors(self):
        with paddle.base.dygraph.guard():
            # The size of b should gather than 2.
            def test_b_size():
                b = paddle.randn([3])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_size)

            # The size of lu should gather than 2.
            def test_lu_size():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_lu_size)

            # The size of pivots should gather than 1.
            def test_pivots_size():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_pivots_size)

            # b.shape[-2] should equal to lu.shape[-2].
            def test_b_lu_shape():
                b = paddle.randn([1, 3])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_lu_shape)

            # lu.shape[-1] should equal to pivots.shape[-1].
            def test_b_pivots_shape():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([2])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_pivots_shape)

            # lu.shape[-2] should equal to lu.shape[-1].
            def test_lu_shape():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 2])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_lu_shape)


if __name__ == "__main__":
    unittest.main()
