#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2022


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False, scale=1.0):
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
    if abs(scale - 1.0) > 1e-09:
        Out = Out * scale
    return Out


class TestMatMulOp(OpTest):
    """
    basic case
    """

    def setUp(self):
        self.set_mlu()
        self.op_type = "matmul"
        self.init_dtype()
        self.init_alpha()
        self.config()

        X = np.random.random(self.x_shape).astype(self.dtype)
        Y = np.random.random(self.y_shape).astype(self.dtype)
        # -0.1 ~ 0.1
        X = -0.1 + 0.2 * X
        Y = -0.1 + 0.2 * Y

        Out = reference_matmul(X, Y, self.transpose_X, self.transpose_Y,
                               self.alpha)
        Out = Out.astype(self.dtype)
        self.inputs = {'X': X, 'Y': Y}
        self.attrs = {
            'transpose_X': self.transpose_X,
            'transpose_Y': self.transpose_Y,
            'alpha': self.alpha
        }
        self.outputs = {'Out': Out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def config(self):
        self.x_shape = (100, )
        self.y_shape = (100, )
        self.transpose_X = False
        self.transpose_Y = False

    def init_alpha(self):
        self.alpha = 1.0

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-7)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')


class TestMatMulOp1(TestMatMulOp):
    """
    case x_ndim == 1, y_ndim != 1
    """

    def config(self):
        self.x_shape = (100, )
        self.y_shape = (1, 3, 2, 100)
        self.transpose_X = False
        self.transpose_Y = True


class TestMatMulOp2(TestMatMulOp):
    """
    case x_ndim != 1, y_ndim == 1
    """

    def config(self):
        self.x_shape = (1, 2, 100, 1)
        self.y_shape = (100, )
        self.transpose_X = True
        self.transpose_Y = False


class TestMatMulOp3(TestMatMulOp):
    """
    case [M, K] x [K, N] = [M, N]
    """

    def config(self):
        self.x_shape = (2, 100)
        self.y_shape = (100, 2)
        self.transpose_X = False
        self.transpose_Y = False


class TestMatMulOp4(TestMatMulOp):
    """
    case [M, K] x [K, N] = [M, N]
    """

    def config(self):
        self.x_shape = (2, 100)
        self.y_shape = (2, 100)
        self.transpose_X = False
        self.transpose_Y = True


class TestMatMulOp5(TestMatMulOp):
    """
    case [M, K] x [K, N] = [M, N]
    """

    def config(self):
        self.x_shape = (100, 2)
        self.y_shape = (100, 2)
        self.transpose_X = True
        self.transpose_Y = False


class TestMatMulOp6(TestMatMulOp):
    """
    case [B, M, K] x [K, N] =  [B, M, N]
    """

    def config(self):
        self.x_shape = (2, 2, 25)
        self.y_shape = (25, 4)
        self.transpose_X = False
        self.transpose_Y = False


class TestMatMulOp7(TestMatMulOp):
    """
    case [B, M, K] x [K, N] =  [B, M, N]
    """

    def config(self):
        self.x_shape = (1, 2, 25)
        self.y_shape = (4, 25)
        self.transpose_X = False
        self.transpose_Y = True


class TestMatMulOp8(TestMatMulOp):
    """
    case [B, M, K] x [K, N] =  [B, M, N]
    """

    def config(self):
        self.x_shape = (1, 25, 4)
        self.y_shape = (25, 4)
        self.transpose_X = True
        self.transpose_Y = False


class TestMatMulOp9(TestMatMulOp):
    """
    case [B, M, K] x  [B, K, N] = [B, M, N]
    """

    def config(self):
        self.x_shape = (2, 5, 10)
        self.y_shape = (2, 10, 5)
        self.transpose_X = False
        self.transpose_Y = False


class TestMatMulOp10(TestMatMulOp):
    """
    case [B, M, K] x  [B, K, N] = [B, M, N]
    """

    def config(self):
        self.x_shape = (2, 10, 5)
        self.y_shape = (2, 10, 5)
        self.transpose_X = True
        self.transpose_Y = False


class TestMatMulOp11(TestMatMulOp):
    """
    case [B, M, K] x  [B, K, N] = [B, M, N]
    """

    def config(self):
        self.x_shape = (2, 5, 10)
        self.y_shape = (2, 5, 10)
        self.transpose_X = False
        self.transpose_Y = True


class TestMatMulOp12(TestMatMulOp):
    """
    case to check the gradient for special case
    """

    def config(self):
        self.x_shape = (100)
        self.y_shape = (1, 2, 2, 100, 2)
        self.transpose_X = False
        self.transpose_Y = False


class TestMatMulOp13(TestMatMulOp):
    """
    case to check the gradient for special case
    """

    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = (100)
        self.transpose_X = False
        self.transpose_Y = False


# TODO(mlu): alpha will be supported in next version
#--------------------test matmul alpha--------------------
# def create_test_alpha_class(parent):
#     class TestMatMulOpAlphaCase(parent):
#         def init_alpha(self):
#             self.alpha = 0.125

#     cls_name = "{0}_{1}".format(parent.__name__, "Alpha")
#     TestMatMulOpAlphaCase.__name__ = cls_name
#     globals()[cls_name] = TestMatMulOpAlphaCase

# create_test_alpha_class(TestMatMulOp)
# create_test_alpha_class(TestMatMulOp1)
# create_test_alpha_class(TestMatMulOp2)
# create_test_alpha_class(TestMatMulOp3)
# create_test_alpha_class(TestMatMulOp4)
# create_test_alpha_class(TestMatMulOp5)
# create_test_alpha_class(TestMatMulOp6)
# create_test_alpha_class(TestMatMulOp9)
# create_test_alpha_class(TestMatMulOp10)
# create_test_alpha_class(TestMatMulOp11)
# create_test_alpha_class(TestMatMulOp12)
# create_test_alpha_class(TestMatMulOp13)


#--------------------test matmul fp16--------------------
def create_test_fp16_class(parent, atol=0.001, max_relative_error=2.5):
    class TestMatMulOpFp16Case(parent):
        def init_kernel_type(self):
            self.dtype = np.float16

        def test_check_output(self):
            self.check_output_with_place(self.place, atol=atol)

        def test_check_grad(self):
            self.check_grad_with_place(
                self.place, ['X', 'Y'],
                'Out',
                max_relative_error=max_relative_error)

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestMatMulOpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestMatMulOpFp16Case


create_test_fp16_class(TestMatMulOp)
create_test_fp16_class(TestMatMulOp1)
create_test_fp16_class(TestMatMulOp2)
create_test_fp16_class(TestMatMulOp3)
create_test_fp16_class(TestMatMulOp4)
create_test_fp16_class(TestMatMulOp5)
create_test_fp16_class(TestMatMulOp6)
create_test_fp16_class(TestMatMulOp9)
create_test_fp16_class(TestMatMulOp10)
create_test_fp16_class(TestMatMulOp11)
create_test_fp16_class(TestMatMulOp12)
create_test_fp16_class(TestMatMulOp13)

if __name__ == "__main__":
    unittest.main()
