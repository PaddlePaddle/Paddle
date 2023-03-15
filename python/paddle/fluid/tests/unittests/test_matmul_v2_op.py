#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import OpTest

import paddle


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
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size,))
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float64")
    return Out


class TestComplexMatMulOp(OpTest):
    def setUp(self):
        self.op_type = "matmul_v2"
        self.python_api = paddle.tensor.matmul
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
        self.x = np.random.random((10, 10)).astype(self.dtype)
        self.y = np.random.random((10, 10)).astype(
            self.dtype
        ) + 1j * np.random.random((10, 10)).astype(self.dtype)
        self.out = np.dot(self.x, self.y)

    def init_grad_input_output(self):
        self.grad_out = np.ones((10, 10), self.dtype) + 1j * np.ones(
            (10, 10), self.dtype
        )
        self.grad_x = np.matmul(self.grad_out, np.conj(self.y).T).real
        self.grad_y = np.matmul(np.conj(self.x).T, self.grad_out)

    # def test_check_output(self):
    #     self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=[self.grad_x, self.grad_y],
            user_defined_grad_outputs=[self.grad_out],
        )

    # def test_check_grad_ingore_x(self):
    #     self.check_grad(
    #         ['Y'],
    #         'Out',
    #         no_grad_set=set("X"),
    #         user_defined_grads=[self.grad_y],
    #         user_defined_grad_outputs=[self.grad_out],
    #     )

    # def test_check_grad_ingore_y(self):
    #     self.check_grad(
    #         ['X'],
    #         'Out',
    #         no_grad_set=set('Y'),
    #         user_defined_grads=[self.grad_x],
    #         user_defined_grad_outputs=[self.grad_out],
    #     )


# class TestComplexMatMulOpBroadcast(OpTest):
#     def setUp(self):
#         self.op_type = "matmul_v2"
#         self.python_api = paddle.tensor.matmul
#         self.init_base_dtype()
#         self.init_input_output()
#         self.init_grad_input_output()

#         self.inputs = {
#             'X': OpTest.np_dtype_to_fluid_dtype(self.x),
#             'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
#         }
#         self.attrs = {'axis': -1, 'use_mkldnn': False}
#         self.outputs = {'Out': self.out}

#     def init_base_dtype(self):
#         self.dtype = np.float64

#     def init_input_output(self):
#         self.x = np.random.random((10, 2, 5)).astype(
#             self.dtype
#         ) + 1j * np.random.random((10, 2, 5)).astype(self.dtype)
#         self.y = np.random.random((5, 20)).astype(
#             self.dtype
#         ) + 1j * np.random.random((5, 20)).astype(self.dtype)
#         self.out = np.dot(self.x, self.y)

#     def init_grad_input_output(self):
#         self.grad_out = np.ones((10, 2, 20), self.dtype) + 1j * np.ones(
#             (10, 2, 20), self.dtype
#         )
#         self.grad_x = np.matmul(self.grad_out, np.conj(self.y).T)
#         self.grad_y = np.sum(
#             np.matmul(np.conj(self.x).transpose(0, 2, 1), self.grad_out), axis=0
#         )

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad_normal(self):
#         self.check_grad(
#             ['X', 'Y'],
#             'Out',
#             user_defined_grads=[self.grad_x, self.grad_y],
#             user_defined_grad_outputs=[self.grad_out],
#         )

#     def test_check_grad_ingore_x(self):
#         self.check_grad(
#             ['Y'],
#             'Out',
#             no_grad_set=set("X"),
#             user_defined_grads=[self.grad_y],
#             user_defined_grad_outputs=[self.grad_out],
#         )

#     def test_check_grad_ingore_y(self):
#         self.check_grad(
#             ['X'],
#             'Out',
#             no_grad_set=set('Y'),
#             user_defined_grads=[self.grad_x],
#             user_defined_grad_outputs=[self.grad_out],
#         )


# class TestMatMulTypePromotion(TestComplexMatMulOp):
#     def init_input_output(self):
#         self.x = np.random.random((10, 10)).astype(self.dtype)
#         self.y = np.random.random((10, 10)).astype(
#             self.dtype
#         ) + 1j * np.random.random((10, 10)).astype(self.dtype)
#         self.out = np.dot(self.x, self.y)

#     def init_grad_input_output(self):
#         self.grad_out = np.ones((10, 10), self.dtype) + 1j * np.ones(
#             (10, 10), self.dtype
#         )
#         self.grad_x = np.matmul(self.grad_out, np.conj(self.y).T).real
#         self.grad_y = np.matmul(np.conj(self.x).T, self.grad_out)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
