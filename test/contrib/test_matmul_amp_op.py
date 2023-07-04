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

import paddle
from paddle.fluid import core
from paddle.fluid.framework import convert_np_dtype_to_dtype_
from paddle.fluid.tests.unittests.eager_op_test import (
    OpTest,
    convert_float_to_uint16,
    get_numeric_gradient,
)
from paddle.fluid.tests.unittests.testsuite import create_op


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size,))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = list(range(len(X.shape)))
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size,))
        else:
            dim = list(range(len(Y.shape)))
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    return Out


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestMatMulAmpOp(OpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = 100
        self.y_shape = (100, 1)
        self.trans_x = False
        self.trans_y = False

    def init_kernel_type(self):
        self.dtype = np.uint16

    def setUp(self):
        self.init_kernel_type()
        self.dx_type = 'bfloat16'
        self.dy_type = 'bfloat16'
        self.config()
        self.op_type = "matmul_amp"
        self.python_api = paddle.incubate.matmul
        if self.is_bfloat16_op():
            x = np.random.random(self.x_shape).astype(np.float32)
            y = np.random.random(self.y_shape).astype(np.float32)
        else:
            x = np.random.random(self.x_shape).astype(self.dtype)
            y = np.random.random(self.y_shape).astype(self.dtype)
            # -0.1 ~ 0.1
            x = -0.1 + 0.2 * x
            y = -0.1 + 0.2 * y
        result = reference_matmul(x, y, self.trans_x, self.trans_y)
        if self.is_bfloat16_op():
            result = result.astype(np.float32)
            self.inputs = {
                'x': convert_float_to_uint16(x),
                'y': convert_float_to_uint16(y),
            }
            self.inputs_fp32 = {
                'x': x,
                'y': y,
            }
        else:
            result = result.astype(self.dtype)
            self.inputs = {
                'x': x,
                'y': y,
            }
        self.attrs = {
            'transpose_x': self.trans_x,
            'transpose_y': self.trans_y,
            'dx_type': convert_np_dtype_to_dtype_(self.dx_type),
            'dy_type': convert_np_dtype_to_dtype_(self.dy_type),
        }
        self.outputs = {'out': result}

    def checker(self, outs):
        np.testing.assert_allclose(
            outs[0], self.outputs['out'], rtol=0.01, atol=0.01
        )

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place_customized(self.checker, place)

    def get_numeric_grad(self, place, check_name):
        scope = core.Scope()
        self._check_grad_helper()
        op = create_op(
            scope, self.op_type, self.inputs_fp32, self.outputs, self.attrs
        )
        return get_numeric_gradient(
            place, scope, op, self.inputs_fp32, check_name, ['out']
        )

    def test_check_grad_x(self):
        place = core.CUDAPlace(0)
        numeric_grads = self.get_numeric_grad(place, 'x')
        self.check_grad_with_place(
            place,
            ['x'],
            'out',
            no_grad_set={'y'},
            user_defined_grads=[numeric_grads],
            check_dygraph=False,
        )

    def test_check_grad_y(self):
        place = core.CUDAPlace(0)
        numeric_grads = self.get_numeric_grad(place, 'y')
        self.check_grad_with_place(
            place,
            ['y'],
            'out',
            no_grad_set={'x'},
            user_defined_grads=[numeric_grads],
            check_dygraph=False,
        )


class TestMatMulAmpOp2(TestMatMulAmpOp):
    """
    case 2
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True
        self.dx_type = 'bfloat16'
        self.dy_type = 'float32'


class TestMatMulAmpOp3(TestMatMulAmpOp):
    """
    case 3
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulAmpOp4(TestMatMulAmpOp):
    """
    case 4
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False
        self.dx_type = 'float32'
        self.dy_type = 'float32'


class TestMatMulAmpOp5(TestMatMulAmpOp):
    """
    case 5
    """

    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100,)
        self.trans_x = True
        self.trans_y = False


class TestMatMulAmpOp6(TestMatMulAmpOp):
    """
    case 6
    """

    def config(self):
        self.x_shape = (1, 2, 102, 1)
        self.y_shape = (102,)
        self.trans_x = True
        self.trans_y = False
        self.dx_type = 'float32'
        self.dy_type = 'bfloat16'


class TestMatMulAmpOp7(TestMatMulAmpOp):
    """
    case 7
    """

    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False
        self.dx_type = 'bfloat16'
        self.dy_type = 'float32'


class TestMatMulAmpOp8(TestMatMulAmpOp):
    """
    case 8
    """

    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False
        self.dx_type = 'float32'
        self.dy_type = 'float32'


class TestMatMulAmpOp9(TestMatMulAmpOp):
    """
    case 9
    """

    def config(self):
        self.x_shape = (1, 1, 1, 100)
        self.y_shape = (2, 1, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulAmpOp10(TestMatMulAmpOp):
    """
    case 10
    """

    def config(self):
        self.x_shape = (1, 1, 25, 4)
        self.y_shape = (1, 2, 4, 25)
        self.trans_x = False
        self.trans_y = False


class TestMatMulAmpOp11(TestMatMulAmpOp):
    """
    case 11
    """

    def config(self):
        self.x_shape = (2, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulAmpOp12(TestMatMulAmpOp):
    """
    case 12
    """

    def config(self):
        self.x_shape = (2, 1, 4, 25)
        self.y_shape = (1, 1, 4, 25)
        self.trans_x = True
        self.trans_y = False


class TestMatMulAmpOp13(TestMatMulAmpOp):
    """
    case 13
    """

    def config(self):
        self.x_shape = (2, 2, 10, 10)
        self.y_shape = (2, 2, 10, 10)
        self.trans_x = True
        self.trans_y = False


class TestMatMulAmpOp14(TestMatMulAmpOp):
    """
    case 14_1
    """

    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = True
        self.trans_y = False


class TestMatMulAmpOp15(TestMatMulAmpOp):
    """
    case 14_2
    """

    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = False
        self.trans_y = False


class TestMatMulAmpOp16(TestMatMulAmpOp):
    """
    case 16 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = 100
        self.y_shape = (1, 2, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulAmpOp17(TestMatMulAmpOp):
    """
    case 17 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = 100
        self.trans_x = False
        self.trans_y = False


class TestMatMulAmpOpBroadcast1(TestMatMulAmpOp):
    """
    case 14_3
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = True
        self.trans_y = True


class TestMatMulAmpOpBroadcast2(TestMatMulAmpOp):
    """
    case 14_4
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatmulAPI(unittest.TestCase):
    def run_test(
        self, x_shape, y_shape, dx_type='bfloat16', dy_type='bfloat16'
    ):
        place = core.CUDAPlace(0)
        input_x = np.random.random(x_shape).astype("float32")
        input_y = np.random.random(y_shape).astype("float32")
        x = paddle.to_tensor(input_x, dtype='bfloat16')
        y = paddle.to_tensor(input_y, dtype='bfloat16')
        x.stop_gradient = False
        y.stop_gradient = False
        result = paddle.paddle.incubate.matmul(
            x, y, False, False, dx_type, dy_type
        )
        dx, dy = paddle.grad([result], [x, y])
        return result, dx, dy

    def test_bf16(self):
        out, dx, dy = self.run_test(x_shape=[1, 100, 4], y_shape=[4, 20])
        self.assertEqual(out.dtype, paddle.float32)
        self.assertEqual(dx.dtype, paddle.bfloat16)
        self.assertEqual(dy.dtype, paddle.bfloat16)

    def test_dx_dy_fp32(self):
        out, dx, dy = self.run_test(
            x_shape=[1, 100, 4],
            y_shape=[4, 20],
            dx_type='float32',
            dy_type='float32',
        )
        self.assertEqual(out.dtype, paddle.float32)
        self.assertEqual(dx.dtype, paddle.float32)
        self.assertEqual(dy.dtype, paddle.float32)

    def test_dx_fp32(self):
        out, dx, dy = self.run_test(
            x_shape=[1, 100, 4],
            y_shape=[4, 20],
            dx_type='float32',
            dy_type='bfloat16',
        )
        self.assertEqual(out.dtype, paddle.float32)
        self.assertEqual(dx.dtype, paddle.float32)
        self.assertEqual(dy.dtype, paddle.bfloat16)

    def test_dy_fp32(self):
        out, dx, dy = self.run_test(
            x_shape=[1, 100, 4],
            y_shape=[4, 20],
            dx_type='bfloat16',
            dy_type='float32',
        )
        self.assertEqual(out.dtype, paddle.float32)
        self.assertEqual(dx.dtype, paddle.bfloat16)
        self.assertEqual(dy.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
