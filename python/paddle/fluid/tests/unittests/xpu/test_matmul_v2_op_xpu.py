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

import sys

sys.path.append("..")
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle.fluid.core as core

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.framework import _test_eager_guard

from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


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


class XPUTestMatmulV2Op(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = "matmul_v2"
        self.use_dynamic_create_class = False

    class TestMatMulV2Op(XPUOpTest):
        """
        case 1
        """

        def config(self):
            self.x_shape = (100, )
            self.y_shape = (100, )
            self.trans_x = False
            self.trans_y = False

        def setUp(self):
            self.dtype = self.in_type
            self.config()
            self.op_type = "matmul_v2"
            if self.dtype == np.float16 or self.dtype == "float16":
                self.__class__.no_need_check_grad = True
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
            if hasattr(self.__class__, "no_need_check_grad"
                       ) and self.__class__.no_need_check_grad == True:
                return
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X', 'Y'], 'Out')

    class TestMatMulOp2(TestMatMulV2Op):
        """
        case 2
        """

        def config(self):
            self.x_shape = (100)
            self.y_shape = (100, 3)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp3(TestMatMulV2Op):
        """
        case 3
        """

        def config(self):
            self.x_shape = (100, )
            self.y_shape = (1, 1, 100, 2)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp4(TestMatMulV2Op):
        """
        case 4
        """

        def config(self):
            self.x_shape = (1, 1, 100, 1)
            self.y_shape = (1, 100)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp5(TestMatMulV2Op):
        """
        case 5
        """

        def config(self):
            self.x_shape = (1, 1, 100, 1)
            self.y_shape = (100, )
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp6(TestMatMulV2Op):
        """
        case 6
        """

        def config(self):
            self.x_shape = (1, 2, 102, 10)
            self.y_shape = (2, 10, 111)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp7(TestMatMulV2Op):
        """
        case 7
        """

        def config(self):
            self.x_shape = (1, 2, 100, 1)
            self.y_shape = (2, 100, 12)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp8(TestMatMulV2Op):
        """
        case 8
        """

        def config(self):
            self.x_shape = (1, 1, 2, 100)
            self.y_shape = (1, 1, 100, 2)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp9(TestMatMulV2Op):
        """
        case 9
        """

        def config(self):
            self.x_shape = (100, 20, 100)
            self.y_shape = (100, 100, 100)
            self.trans_x = False
            self.trans_y = True

    class TestMatMulOp10(TestMatMulV2Op):
        """
        case 10
        """

        def config(self):
            self.x_shape = (100, 20, 100)
            self.y_shape = (100, 20, 100)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp11(TestMatMulV2Op):
        """
        case 11
        """

        def config(self):
            self.x_shape = (2, 20, 100)
            self.y_shape = (100, 30)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp12(TestMatMulV2Op):
        """
        case 12
        """

        def config(self):
            self.x_shape = (1, 20, 100)
            self.y_shape = (100, )
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp13(TestMatMulV2Op):
        """
        case 13
        """

        def config(self):
            self.x_shape = (2, 2, 10, 10)
            self.y_shape = (2, 2, 10, 10)
            self.trans_x = True
            self.trans_y = False

    class TestMatMulOp14(TestMatMulV2Op):
        """
        case 14_1
        """

        def config(self):
            self.x_shape = (100, 2, 100, 10)
            self.y_shape = (100, 2, 10, 90)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp15(TestMatMulV2Op):
        """
        case 14_2
        """

        def config(self):
            self.x_shape = (100, 2, 100, 10)
            self.y_shape = (100, 2, 100, 10)
            self.trans_x = False
            self.trans_y = True

    class TestMatMulOp16(TestMatMulV2Op):
        """
        case 16 : to check the big data
        """

        def config(self):
            self.x_shape = (1000, 2, 100, 100)
            self.y_shape = (1000, 2, 100, 900)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp17(TestMatMulV2Op):
        """
        case 17 : to check the gradient for special case
        """

        def config(self):
            self.x_shape = (2, 1, 100)
            self.y_shape = (100)
            self.trans_x = False
            self.trans_y = False

    class TestMatMulOp18(TestMatMulV2Op):
        """
        case 18 : for ppyoloe model
        """

        def config(self):
            self.x_shape = (8, 111, 4, 17)
            self.y_shape = (17)
            self.trans_x = False
            self.trans_y = False


support_types = get_xpu_op_support_types('matmul_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestMatmulV2Op, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
