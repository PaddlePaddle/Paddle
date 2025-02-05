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
from op_test import OpTest, convert_float_to_uint16

import paddle
import paddle.base.dygraph as dg
from paddle import base
from paddle.base import core


class TestKronOp(OpTest):
    def setUp(self):
        self.op_type = "kron"
        self.prim_op_type = "prim"
        self.python_api = paddle.kron
        self.public_python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(10, 10)).astype(self.dtype)
        y = np.random.uniform(size=(10, 10)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}

    def _init_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set('X'),
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_pir=True,
            check_prim_pir=True,
        )


class TestKronOp2(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.prim_op_type = "prim"
        self.python_api = paddle.kron
        self.public_python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(5, 5, 4)).astype(self.dtype)
        y = np.random.uniform(size=(100)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronOp3(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.prim_op_type = "prim"
        self.python_api = paddle.kron
        self.public_python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(10, 10)).astype(self.dtype)
        y = np.random.uniform(size=(5, 5, 4)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronOp4(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.prim_op_type = "prim"
        self.python_api = paddle.kron
        self.public_python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(5, 5, 5)).astype(self.dtype)
        y = np.random.uniform(size=(2, 3, 4, 2, 2, 3)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronOp5(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.prim_op_type = "prim"
        self.python_api = paddle.kron
        self.public_python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(2, 3, 4, 3, 2)).astype(self.dtype)
        y = np.random.uniform(size=(10, 10)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronOp6(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.prim_op_type = "prim"
        self.python_api = paddle.kron
        self.public_python_api = paddle.kron
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(10, 10, 2)).astype(self.dtype)
        y = np.random.uniform(size=(2, 3, 4, 3, 2)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronFP16Op(TestKronOp):
    def _init_dtype(self):
        return "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestKronBF16Op(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.prim_op_type = "prim"
        self.python_api = paddle.kron
        self.public_python_api = paddle.kron
        self.dtype = np.uint16
        self.np_dtype = "float32"
        x = np.random.uniform(size=(10, 10)).astype(self.np_dtype)
        y = np.random.uniform(size=(10, 10)).astype(self.np_dtype)
        out_ref = np.kron(x, y)
        self.inputs = {
            'X': convert_float_to_uint16(x),
            'Y': convert_float_to_uint16(y),
        }
        self.outputs = {'Out': convert_float_to_uint16(out_ref)}
        # bfloat16 requires using place
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ['X', 'Y'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            no_grad_set=set('X'),
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_pir=True,
            check_prim_pir=True,
        )


class TestKronLayer(unittest.TestCase):
    def test_case(self):
        a = np.random.randn(10, 10).astype(np.float64)
        b = np.random.randn(10, 10).astype(np.float64)
        place = base.CPUPlace()
        with dg.guard(place):
            a_var = paddle.to_tensor(a)
            b_var = paddle.to_tensor(b)
            c_var = paddle.kron(a_var, b_var)
            np.testing.assert_allclose(c_var.numpy(), np.kron(a, b))

    def test_case_with_output(self):
        place = base.CPUPlace()
        a = np.random.randn(10, 10).astype(np.float64)
        b = np.random.randn(10, 10).astype(np.float64)
        out_np = np.kron(a, b)
        paddle.enable_static()
        prog = paddle.static.Program()
        with base.unique_name.guard():
            with paddle.static.program_guard(prog, prog):
                a_var = paddle.static.data("a", [-1, -1], dtype="float64")
                b_var = paddle.static.data("b", [-1, -1], dtype="float64")
                out_var = paddle.kron(a_var, b_var)
        exe = paddle.static.Executor(place=place)
        (res,) = exe.run(prog, feed={'a': a, 'b': b}, fetch_list=[out_var])
        np.testing.assert_allclose(res, out_np)


class TestComplexKronOp(OpTest):
    def setUp(self):
        self.op_type = "kron"
        self.python_api = paddle.kron
        self.x_shape = np.array([10, 10])
        self.y_shape = np.array([3, 35])
        self.out_shape = self.x_shape * self.y_shape
        self.init_base_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.complex128

    def init_input_output(self):
        self.x = np.random.random(self.x_shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.y_shape).astype(self.dtype)
        self.out = np.kron(self.x, self.y)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_pir=True,
        )

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_pir=True,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_pir=True,
        )


class TestKronOpTypePromotion(TestComplexKronOp):
    def init_input_output(self):
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.y_shape).astype(self.dtype)
        self.out = np.kron(self.x, self.y)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
