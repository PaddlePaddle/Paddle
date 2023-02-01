# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg


class TestKronOp(OpTest):
    def setUp(self):
        self.op_type = "kron"
        self.python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(10, 10)).astype(self.dtype)
        y = np.random.uniform(size=(10, 10)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}

    def _init_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_eager=True)

    def test_check_grad_ignore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set('X'), check_eager=True)

    def test_check_grad_ignore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'), check_eager=True)


class TestKronOp2(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(5, 5, 4)).astype(self.dtype)
        y = np.random.uniform(size=(10, 10)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronOp3(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(10, 10)).astype(self.dtype)
        y = np.random.uniform(size=(5, 5, 4)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronLayer(unittest.TestCase):
    def test_case(self):
        a = np.random.randn(10, 10).astype(np.float64)
        b = np.random.randn(10, 10).astype(np.float64)

        place = fluid.CPUPlace()
        with dg.guard(place):
            a_var = dg.to_variable(a)
            b_var = dg.to_variable(b)
            c_var = paddle.kron(a_var, b_var)
            np.testing.assert_allclose(c_var.numpy(), np.kron(a, b))

    def test_case_with_output(self):
        a = np.random.randn(10, 10).astype(np.float64)
        b = np.random.randn(10, 10).astype(np.float64)

        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                a_var = fluid.data("a", [-1, -1], dtype="float64")
                b_var = fluid.data("b", [-1, -1], dtype="float64")
                out_var = paddle.kron(a_var, b_var)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(start)
        (c,) = exe.run(main, feed={'a': a, 'b': b}, fetch_list=[out_var])
        np.testing.assert_allclose(c, np.kron(a, b))


class TestComplexKronOp(OpTest):
    def setUp(self):
        self.op_type = "kron"
        self.python_api = paddle.kron
        self.x_shape = np.array([10, 10])
        self.y_shape = np.array([3, 35])
        self.out_shape = self.x_shape * self.y_shape
        self.init_base_dtype()
        self.init_input_output()
        self.init_grad_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.float64

    def init_input_output(self):
        self.x = np.random.random(self.x_shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.y_shape).astype(self.dtype)
        self.out = np.kron(self.x, self.y)

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.out_shape, self.dtype) + 1j * np.ones(
            self.out_shape, self.dtype
        )
        self.grad_x = self.get_grad_x_by_numpy()
        self.grad_y = self.get_grad_y_by_numpy()

    def get_grad_x_by_numpy(self):
        grad_x = np.zeros(self.x_shape, np.complex128)
        for x_i in range(self.x_shape[0]):
            for x_j in range(self.x_shape[1]):
                for i in range(self.y_shape[0]):
                    for j in range(self.y_shape[1]):
                        idx_i = x_i * self.y_shape[0] + i
                        idx_j = x_j * self.y_shape[1] + j
                        grad_x[x_i][x_j] += self.grad_out[idx_i][
                            idx_j
                        ] * np.conj(self.y[i][j])
        return grad_x

    def get_grad_y_by_numpy(self):
        grad_y = np.zeros(self.y_shape, np.complex128)
        for y_i in range(self.y_shape[0]):
            for y_j in range(self.y_shape[1]):
                for x_i in range(self.x_shape[0]):
                    for x_j in range(self.x_shape[1]):
                        idx_i = x_i * self.y_shape[0] + y_i
                        idx_j = x_j * self.y_shape[1] + y_j
                        grad_y[y_i][y_j] += self.grad_out[idx_i][
                            idx_j
                        ] * np.conj(self.x[x_i][x_j])
        return grad_y

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=[self.grad_x, self.grad_y],
            user_defined_grad_outputs=[self.grad_out],
            check_eager=True,
        )

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            user_defined_grads=[self.grad_y],
            user_defined_grad_outputs=[self.grad_out],
            check_eager=True,
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out],
            check_eager=True,
        )


class TestKronOpTypePromotion(TestComplexKronOp):
    def init_input_output(self):
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.y_shape).astype(self.dtype)
        self.out = np.kron(self.x, self.y)

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.out_shape, self.dtype) + 1j * np.ones(
            self.out_shape, self.dtype
        )
        self.grad_x = self.get_grad_x_by_numpy().real
        self.grad_y = self.get_grad_y_by_numpy()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
