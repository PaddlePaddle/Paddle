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
        self.x_shape = (100, 4)
        self.y_shape = (4, 100)
        self.trans_x = False
        self.trans_y = False

    def init_kernel_type(self):
        self.dtype = np.uint16

    def setUp(self):
        self.init_kernel_type()
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
        self.attrs = {'trans_x': self.trans_x, 'trans_y': self.trans_y}
        self.outputs = {'out': result}

    def checker(self, outs):
        np.testing.assert_allclose(outs[0], self.outputs['out'])

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place_customized(
            checker=self.checker, place=place
        )

    def get_numeric_grad(self, place, check_name):
        scope = core.Scope()
        self._check_grad_helper()
        op = create_op(
            scope, self.op_type, self.inputs, self.outputs, self.attrs
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
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['x', 'y'], 'out')


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
