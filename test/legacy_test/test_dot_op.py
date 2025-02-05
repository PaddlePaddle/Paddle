#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
from paddle.base import core


class DotOp(OpTest):
    def setUp(self):
        self.op_type = "dot"
        self.prim_op_type = "prim"
        self.python_api = paddle.dot
        self.public_python_api = paddle.dot
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {'Out': self.out}
        self.attrs = {}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            if core.is_compiled_with_rocm():
                self.check_grad(
                    ['X', 'Y'],
                    'Out',
                    user_defined_grads=[self.inputs['Y'], self.inputs['X']],
                    check_pir=True,
                )
            else:
                self.check_grad(['X', 'Y'], 'Out', check_pir=True)
        else:
            if core.is_compiled_with_rocm():
                self.check_grad(
                    ['X', 'Y'],
                    'Out',
                    user_defined_grads=[self.inputs['Y'], self.inputs['X']],
                    check_pir=True,
                )
            else:
                self.check_grad(
                    ['X', 'Y'], 'Out', check_pir=True, check_prim_pir=True
                )

    def test_check_grad_ignore_x(self):
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            if core.is_compiled_with_rocm():
                self.check_grad(
                    ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    user_defined_grads=[self.inputs['X']],
                    check_pir=True,
                )
            else:
                self.check_grad(
                    ['Y'], 'Out', no_grad_set=set("X"), check_pir=True
                )
        else:
            if core.is_compiled_with_rocm():
                self.check_grad(
                    ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    user_defined_grads=[self.inputs['X']],
                    check_pir=True,
                )
            else:
                self.check_grad(
                    ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    check_pir=True,
                    check_prim_pir=True,
                )

    def test_check_grad_ignore_y(self):
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            if core.is_compiled_with_rocm():
                self.check_grad(
                    ['X'],
                    'Out',
                    no_grad_set=set('Y'),
                    user_defined_grads=[self.inputs['Y']],
                    check_pir=True,
                )
            else:
                self.check_grad(
                    ['X'], 'Out', no_grad_set=set('Y'), check_pir=True
                )
        else:
            if core.is_compiled_with_rocm():
                self.check_grad(
                    ['X'],
                    'Out',
                    no_grad_set=set('Y'),
                    user_defined_grads=[self.inputs['Y']],
                    check_pir=True,
                )
            else:
                self.check_grad(
                    ['X'],
                    'Out',
                    no_grad_set=set('Y'),
                    check_pir=True,
                    check_prim_pir=True,
                )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [121]).astype(self.dtype)
        self.y = np.random.uniform(1, 3, [121]).astype(self.dtype)
        self.out = np.dot(self.x, self.y).astype(self.dtype)

    def init_dtype(self):
        self.dtype = np.float64


class DotOpBatch(DotOp):
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
        self.check_grad(['X', 'Y'], 'Out', check_pir=True, check_prim_pir=True)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
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


class TestDotOpError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # the input dtype of elementwise_mul must be float16 or float32 or float64 or int32 or int64
            # float16 only can be set on GPU place
            x1 = paddle.static.data(name='x1', shape=[-1, 120], dtype="uint8")
            y1 = paddle.static.data(name='y1', shape=[-1, 120], dtype="uint8")
            self.assertRaises(Exception, paddle.dot, x1, y1)

            x2 = paddle.static.data(
                name='x2', shape=[-1, 2, 3], dtype="float32"
            )
            y2 = paddle.static.data(
                name='y2', shape=[-1, 2, 3], dtype="float32"
            )
            self.assertRaises(Exception, paddle.dot, x2, y2)

            x3 = paddle.static.data(name='x3', shape=[-1, 3], dtype="float32")
            y3 = paddle.static.data(
                name='y3', shape=[-1, 2, 3], dtype="float32"
            )
            self.assertRaises(Exception, paddle.dot, x2, y3)


class TestDygraph(unittest.TestCase):
    def test_dygraph(self):
        with base.dygraph.guard():
            x1 = paddle.to_tensor(np.array([1, 3]).astype(np.float32))
            y1 = paddle.to_tensor(np.array([2, 5]).astype(np.float32))
            np.testing.assert_allclose(
                paddle.dot(x1, y1).numpy(), np.array([17]), rtol=1e-05
            )

            x1 = paddle.to_tensor(np.array([[1, 3], [3, 5]]).astype(np.float32))
            y1 = paddle.to_tensor(np.array([[2, 5], [6, 8]]).astype(np.float32))
            np.testing.assert_array_equal(
                paddle.dot(x1, y1).numpy(), np.array([17, 58])
            )


class TestComplex64DotOp(DotOp):
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
        self.out = np.dot(self.x, self.y).astype(self.dtype)


class TestComplex64DotOp2D(TestComplex64DotOp):
    def init_input_output(self):
        shape = (2, 100)
        self.x = (
            np.random.random(shape) + 1j * np.random.random(shape)
        ).astype(self.dtype)
        self.y = (
            np.random.random(shape) + 1j * np.random.random(shape)
        ).astype(self.dtype)
        self.out = np.diag(np.dot(self.x, self.y.T)).reshape(-1)


class TestComplex128DotOp(TestComplex64DotOp):
    def init_dtype(self):
        self.dtype = np.complex128


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestDotFP16Op(OpTest):
    def setUp(self):
        self.op_type = "dot"
        self.python_api = paddle.dot
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
                self.check_output_with_place(place, atol=0.125, check_pir=True)

    def test_check_grad_normal(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place, ['X', 'Y'], 'Out', check_pir=True
                )

    def test_check_grad_ignore_x(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place, ['Y'], 'Out', no_grad_set=set("X"), check_pir=True
                )

    def test_check_grad_ignore_y(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place, ['X'], 'Out', no_grad_set=set("Y"), check_pir=True
                )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [121]).astype(self.dtype)
        self.y = np.random.uniform(1, 3, [121]).astype(self.dtype)
        self.out = np.dot(self.x, self.y)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class DotFP16OpBatch(TestDotFP16Op):
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


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestDotBF16Op(OpTest):
    def setUp(self):
        self.op_type = "dot"
        self.python_api = paddle.dot
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
                self.check_output_with_place(place, atol=0.5, check_pir=True)

    def test_check_grad_normal(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['X', 'Y'],
                    'Out',
                    user_defined_grads=[self.inputs['Y'], self.inputs['X']],
                    check_pir=True,
                )

    def test_check_grad_ignore_x(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    user_defined_grads=[self.inputs['X']],
                    check_pir=True,
                )

    def test_check_grad_ignore_y(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    no_grad_set=set("Y"),
                    user_defined_grads=[self.inputs['Y']],
                    check_pir=True,
                )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [121]).astype(np.float32)
        self.y = np.random.uniform(1, 3, [121]).astype(np.float32)
        self.out = np.dot(self.x, self.y)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class DotBF16OpBatch(TestDotBF16Op):
    def init_input_output(self):
        self.x = (
            np.random.uniform(0.1, 1, [132])
            .astype(np.float32)
            .reshape([11, 12])
        )
        self.y = (
            np.random.uniform(1, 3, [132]).astype(np.float32).reshape([11, 12])
        )
        self.out = np.sum(self.x * self.y, axis=1)

    def test_check_grad_normal(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['X', 'Y'],
                    'Out',
                    user_defined_grads=[
                        self.y / self.y.shape[0],
                        self.x / self.x.shape[0],
                    ],
                    check_pir=True,
                )

    def test_check_grad_ignore_x(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    user_defined_grads=[self.x / self.x.shape[0]],
                    check_pir=True,
                )

    def test_check_grad_ignore_y(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    no_grad_set=set("Y"),
                    user_defined_grads=[self.y / self.y.shape[0]],
                    check_pir=True,
                )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
