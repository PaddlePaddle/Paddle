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

from __future__ import print_function
import sys
sys.path.append("..")
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle.fluid.core as core

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size, ))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size, ))
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


class TestMatMulV2Op(XPUOpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = (100, )
        self.y_shape = (100, )
        self.trans_x = False
        self.trans_y = False

    def init_kernel_type(self):
        self.dtype = "float32"

    def setUp(self):
        self.use_xpu = True
        self.init_kernel_type()
        self.config()
        self.op_type = "matmul_v2"
        x = np.random.random(self.x_shape).astype(self.dtype)
        y = np.random.random(self.y_shape).astype(self.dtype)
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
        result = reference_matmul(x, y, self.trans_x, self.trans_y)
        result = result.astype(self.dtype)
        self.inputs = {
            'X': x,
            'Y': y,
        }
        self.attrs = {'trans_x': self.trans_x, 'trans_y': self.trans_y}
        self.outputs = {'Out': result}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        place = paddle.XPUPlace(0)
        self.check_grad_with_place(place, ['X', 'Y'], 'Out')


# class TestMatMuklOp2(TestMatMulV2Op):
#     """
#     case 2
#     """

#     def config(self):
#         self.x_shape = (100, )
#         self.y_shape = (1, 3, 2, 100)
#         self.trans_x = False
#         self.trans_y = True


class TestMatMuklOp3(TestMatMulV2Op):
    """
    case 3
    """

    def config(self):
        self.x_shape = (100, )
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


# class TestMatMuklOp4(TestMatMulV2Op):
#     """
#     case 4
#     """

#     def config(self):
#         self.x_shape = (100, )
#         self.y_shape = (1, 2, 100, 2)
#         self.trans_x = False
#         self.trans_y = False


class TestMatMuklOp5(TestMatMulV2Op):
    """
    case 5
    """

    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100, )
        self.trans_x = True
        self.trans_y = False


# class TestMatMuklOp6(TestMatMulV2Op):
#     """
#     case 6
#     """

#     def config(self):
#         self.x_shape = (1, 2, 102, 1)
#         self.y_shape = (102, )
#         self.trans_x = True
#         self.trans_y = False

# class TestMatMuklOp7(TestMatMulV2Op):
#     """
#     case 7
#     """

#     def config(self):
#         self.x_shape = (1, 2, 1, 100)
#         self.y_shape = (100, )
#         self.trans_x = False
#         self.trans_y = False


class TestMatMuklOp8(TestMatMulV2Op):
    """
    case 8
    """

    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


# class TestMatMuklOp9(TestMatMulV2Op):
#     """
#     case 9
#     """

#     def config(self):
#         self.x_shape = (1, 1, 1, 100)
#         self.y_shape = (2, 1, 2, 100)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatMuklOp10(TestMatMulV2Op):
#     """
#     case 10
#     """

#     def config(self):
#         self.x_shape = (1, 1, 25, 4)
#         self.y_shape = (1, 2, 4, 25)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatMuklOp11(TestMatMulV2Op):
#     """
#     case 11
#     """

#     def config(self):
#         self.x_shape = (2, 1, 2, 100)
#         self.y_shape = (1, 1, 100, 2)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatMuklOp12(TestMatMulV2Op):
#     """
#     case 12
#     """

#     def config(self):
#         self.x_shape = (2, 1, 4, 25)
#         self.y_shape = (1, 1, 4, 25)
#         self.trans_x = True
#         self.trans_y = False


class TestMatMuklOp13(TestMatMulV2Op):
    """
    case 13
    """

    def config(self):
        self.x_shape = (2, 2, 10, 10)
        self.y_shape = (2, 2, 10, 10)
        self.trans_x = True
        self.trans_y = False


# class TestMatMuklOp14(TestMatMulV2Op):
#     """
#     case 14_1
#     """

#     def config(self):
#         self.x_shape = (3, 1, 6, 6)
#         self.y_shape = (1, 2, 6, 9)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatMuklOp15(TestMatMulV2Op):
#     """
#     case 14_2
#     """

#     def config(self):
#         self.x_shape = (3, 1, 6, 6)
#         self.y_shape = (1, 2, 6, 9)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatMuklOp16(TestMatMulV2Op):
#     """
#     case 16 : to check the gradient for special case
#     """

#     def config(self):
#         self.x_shape = (100)
#         self.y_shape = (1, 2, 2, 100, 2)
#         self.trans_x = False
#         self.trans_y = False


class TestMatMuklOp17(TestMatMulV2Op):
    """
    case 17 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = (100)
        self.trans_x = False
        self.trans_y = False


# class TestMatMuklOpBroadcast1(TestMatMulV2Op):
#     """
#     case 14_3
#     """

#     def config(self):
#         self.x_shape = (3, 1, 10, 10)
#         self.y_shape = (1, 2, 10, 10)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatMuklOpBroadcast2(TestMatMulV2Op):
#     """
#     case 14_4
#     """

#     def config(self):
#         self.x_shape = (3, 1, 10, 10)
#         self.y_shape = (1, 2, 10, 10)
#         self.trans_x = False
#         self.trans_y = True

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
