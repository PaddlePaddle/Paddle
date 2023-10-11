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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


class VdotOp(OpTest):
    def setUp(self):
        self.op_type = "vdot"
        self.python_api = paddle.vdot
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {'Out': self.out}
        self.attrs = {}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
        )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [121]).astype(self.dtype)
        self.y = np.random.uniform(1, 3, [121]).astype(self.dtype)
        self.out = np.vdot(self.x, self.y).astype(self.dtype)

    def init_dtype(self):
        self.dtype = np.float64


class VdotOpBatch(VdotOp):
    def init_input_output(self):
        self.x = (
            np.random.uniform(0.1, 1, [132])
            .astype(self.dtype)
            .reshape([11, 12])
        )
        self.y = (
            np.random.uniform(1, 3, [132]).astype(self.dtype).reshape([11, 12])
        )
        self.out = np.sum(self.x * self.y, axis=1)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestDygraph(unittest.TestCase):
    def test_dygraph(self):
        with base.dygraph.guard():
            for dtype in [np.int32, np.int64, np.float32]:
                x1 = base.dygraph.to_variable(np.array([1, 3]).astype(dtype))
                y1 = base.dygraph.to_variable(np.array([2, 5]).astype(dtype))
                np.testing.assert_allclose(
                    paddle.vdot(x1, y1).numpy(), np.array([17]), rtol=1e-05
                )


class TestComplex64VdotOp(VdotOp):
    def init_dtype(self):
        self.dtype = np.complex64

    def init_input_output(self):
        shape = 100
        self.x = (
            np.random.random(shape) + 1j * np.random.random(shape)
        ).astype(self.dtype)
        self.y = (
            np.random.random(shape) + 1j * np.random.random(shape)
        ).astype(self.dtype)
        self.out = np.vdot(self.x, self.y).astype(self.dtype)


class TestComplex128VdotOp(TestComplex64VdotOp):
    def init_dtype(self):
        self.dtype = np.complex128


class TestComplex64VdotOp2D(TestComplex64VdotOp):
    def init_input_output(self):
        shape = (3, 100)
        self.x = (
            np.random.random(shape) + 1j * np.random.random(shape)
        ).astype(self.dtype)
        self.y = (
            np.random.random(shape) + 1j * np.random.random(shape)
        ).astype(self.dtype)
        self.out = np.array(
            [np.vdot(t[0], t[1]) for t in list(zip(self.x, self.y))]
        ).astype(self.dtype)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestComplex128VdotOp2D(TestComplex128VdotOp):
    def init_input_output(self):
        shape = (3, 100)
        self.x = (
            np.random.random(shape) + 1j * np.random.random(shape)
        ).astype(self.dtype)
        self.y = (
            np.random.random(shape) + 1j * np.random.random(shape)
        ).astype(self.dtype)
        self.out = np.array(
            [np.vdot(t[0], t[1]) for t in list(zip(self.x, self.y))]
        ).astype(self.dtype)


class VdotOpEmptyInput(unittest.TestCase):
    def test_1d_input(self):
        data = np.array([], dtype=np.float32)
        x = paddle.to_tensor(np.reshape(data, [0]), dtype='float32')
        y = paddle.to_tensor(np.reshape(data, [0]), dtype='float32')
        np_out = np.vdot(data, data)
        pd_out = paddle.vdot(x, y)
        self.assertEqual(np_out, pd_out)


class TestVdotOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input dtype of elementwise_mul must be float16 or float32 or float64 or int32 or int64
            # float16 only can be set on GPU place
            x1 = paddle.static.data(name='x1', shape=[-1, 120], dtype="uint8")
            y1 = paddle.static.data(name='y1', shape=[-1, 120], dtype="uint8")
            self.assertRaises(Exception, paddle.vdot, x1, y1)

            x2 = paddle.static.data(
                name='x2', shape=[-1, 2, 3], dtype="float32"
            )
            y2 = paddle.static.data(
                name='y2', shape=[-1, 2, 3], dtype="float32"
            )
            self.assertRaises(Exception, paddle.vdot, x2, y2)

            x3 = paddle.static.data(name='x3', shape=[-1, 3], dtype="float32")
            y3 = paddle.static.data(
                name='y3', shape=[-1, 2, 3], dtype="float32"
            )
            self.assertRaises(Exception, paddle.vdot, x2, y3)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestVdotFP16Op(OpTest):
    def setUp(self):
        self.op_type = "vdot"
        self.python_api = paddle.vdot
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {'Out': self.out}
        self.attrs = {}

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=0.125)

    def test_check_grad_normal(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(place, ['X', 'Y'], 'Out')

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [121]).astype(self.dtype)
        self.y = np.random.uniform(1, 3, [121]).astype(self.dtype)
        self.out = np.vdot(self.x, self.y)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestVdotBF16Op(OpTest):
    def setUp(self):
        self.op_type = "vdot"
        self.python_api = paddle.vdot
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': convert_float_to_uint16(self.x),
            'Y': convert_float_to_uint16(self.y),
        }
        self.outputs = {'Out': convert_float_to_uint16(self.out)}
        self.attrs = {}

    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_output_with_place(place, atol=0.5)

    def test_check_grad_normal(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['X', 'Y'],
                    'Out',
                    user_defined_grads=[self.inputs['Y'], self.inputs['X']],
                )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [121]).astype(np.float32)
        self.y = np.random.uniform(1, 3, [121]).astype(np.float32)
        self.out = np.vdot(self.x, self.y)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
