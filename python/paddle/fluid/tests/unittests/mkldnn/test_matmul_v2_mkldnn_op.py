#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.mkldnn.test_matmul_mkldnn_op import (
    TestMatMulOpTransposeReshapeBasicFloat,
    TestMatMulOpTransposeReshapeEmptyFloat,
    TestMatMulOpTransposeReshapeOtherDimFloat,
    TestReshapeTransposeMatMulOp,
    TestReshapeTransposeMatMulOp2DXFloat,
    TestReshapeTransposeMatMulOp2DYFloat,
    TestReshapeTransposeMatMulOp3DXFloat,
    TestReshapeTransposeMatMulOp3DYFloat,
    TestReshapeTransposeMatMulOp4DXFloat,
    TestReshapeTransposeMatMulOp4DXYFloat,
    TestReshapeTransposeMatMulOp4DYFloat,
)
from paddle.fluid.tests.unittests.op_test import (
    OpTest,
    OpTestTool,
    convert_float_to_uint16,
)
=======
from __future__ import print_function

import unittest
from functools import reduce
import numpy as np

from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle.fluid.core as core
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.tests.unittests.mkldnn.test_matmul_mkldnn_op import (
    TestMatMulOpTransposeReshapeEmptyFloat,
    TestMatMulOpTransposeReshapeBasicFloat,
    TestMatMulOpTransposeReshapeOtherDimFloat, TestReshapeTransposeMatMulOp,
    TestReshapeTransposeMatMulOp4DXFloat, TestReshapeTransposeMatMulOp4DYFloat,
    TestReshapeTransposeMatMulOp4DXYFloat, TestReshapeTransposeMatMulOp2DXFloat,
    TestReshapeTransposeMatMulOp2DYFloat, TestReshapeTransposeMatMulOp3DXFloat,
    TestReshapeTransposeMatMulOp3DYFloat)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def reference_matmul(X, Y, transpose_x=False, transpose_y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_x:
        if X.ndim == 1:
<<<<<<< HEAD
            X = X.reshape((X.size,))
=======
            X = X.reshape((X.size, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_y:
        if Y.ndim == 1:
<<<<<<< HEAD
            Y = Y.reshape((Y.size,))
=======
            Y = Y.reshape((Y.size, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.atleast_1d(np.matmul(X, Y))
    return Out


class TestMatMulV2VectorXVectorOneDNNOp(OpTest):
<<<<<<< HEAD
    def config(self):
        self.x_shape = (100,)
        self.y_shape = (100,)
=======

    def config(self):
        self.x_shape = (100, )
        self.y_shape = (100, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.trans_x = False
        self.trans_y = False
        self._cpu_only = True
        self.use_mkldnn = True

    def set_inputs(self, x, y):
        self.inputs = {'X': x, 'Y': y}

    def set_dtype_attr(self):
        self.attrs['mkldnn_data_type'] = "float32"

    def setUp(self):
        self.config()
        self.op_type = "matmul_v2"
        x = np.random.random(self.x_shape).astype("float32")
        y = np.random.random(self.y_shape).astype("float32")
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
<<<<<<< HEAD
        result = reference_matmul(x, y, self.trans_x, self.trans_y).astype(
            "float32"
        )
=======
        result = reference_matmul(x, y, self.trans_x,
                                  self.trans_y).astype("float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.set_inputs(x, y)
        self.attrs = {
            'trans_x': self.trans_x,
            'trans_y': self.trans_y,
<<<<<<< HEAD
            'use_mkldnn': True,
=======
            'use_mkldnn': True
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.set_dtype_attr()
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestMatMulV2VectorXMatrixTransposeYOneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
    def config(self):
        self.x_shape = (100,)
=======
        TestMatMulV2VectorXVectorOneDNNOp):

    def config(self):
        self.x_shape = (100, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2VectorXMatrixOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
    def config(self):
        self.x_shape = (100,)
=======

    def config(self):
        self.x_shape = (100, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXVectorTransposeXOneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100,)
=======
        TestMatMulV2VectorXVectorOneDNNOp):

    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXVectorOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100,)
=======

    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 1)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeYOneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (1, 1, 1, 100)
        self.y_shape = (2, 1, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2MatrixXMatrix2OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (2, 1, 12, 9)
        self.y_shape = (1, 3, 9, 12)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrix3OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (2, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTranposeXOneDNNOp2(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (2, 1, 4, 25)
        self.y_shape = (1, 1, 4, 25)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTranposeX2OneDNNOp3(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (2, 2, 7, 4)
        self.y_shape = (2, 2, 7, 5)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeX3OneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (3, 1, 6, 7)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrix4OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2VectorXMatrix5DOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
    def config(self):
        self.x_shape = 100
=======

    def config(self):
        self.x_shape = (100)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.y_shape = (1, 2, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2Matrix3DXVectorOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = 100
=======

    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = (100)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeXTransposeYOneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (3, 1, 10, 8)
        self.y_shape = (1, 2, 9, 10)
        self.trans_x = True
        self.trans_y = True


class TestMatMulV2MatrixXMatrixTransposeY2OneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 9, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2MatrixXMatrix5DTranposeYOneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (1, 3, 1, 10, 10)
        self.y_shape = (3, 1, 2, 9, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2MatrixXMatrix6Dx2DOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (1, 1, 2, 1, 8, 9)
        self.y_shape = (9, 12)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrix2Dx5DOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (20, 5)
        self.y_shape = (1, 2, 1, 5, 11)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrix4Dx3DTransposeXOneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (5, 4, 15, 10)
        self.y_shape = (1, 15, 20)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrix3Dx4DTransposeYOneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (2, 10, 15)
        self.y_shape = (4, 2, 20, 15)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2MatrixXMatrix5Dx3DTransposeXTransposeYOneDNNOp(
<<<<<<< HEAD
    TestMatMulV2VectorXVectorOneDNNOp
):
=======
        TestMatMulV2VectorXVectorOneDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (4, 3, 2, 15, 10)
        self.y_shape = (1, 20, 15)
        self.trans_x = True
        self.trans_y = True


class TestMatMulV2MatrixXMatrix3Dx4DOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.x_shape = (1, 1, 32, 16)
        self.y_shape = (16, 16, 16)
        self.trans_x = False
        self.trans_y = False


#   BF16 TESTS
def create_bf16_test_class(parent):
<<<<<<< HEAD
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestMatMulV2Bf16OneDNNOp(parent):
        def set_inputs(self, x, y):
            self.inputs = {
                'X': convert_float_to_uint16(x),
                'Y': convert_float_to_uint16(y),
=======

    @OpTestTool.skip_if_not_cpu_bf16()
    class TestMatMulV2Bf16OneDNNOp(parent):

        def set_inputs(self, x, y):
            self.inputs = {
                'X': convert_float_to_uint16(x),
                'Y': convert_float_to_uint16(y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.x_fp32 = x
            self.y_fp32 = y

        def set_dtype_attr(self):
            self.attrs['mkldnn_data_type'] = "bfloat16"

        def test_check_output(self):
            self.check_output_with_place(core.CPUPlace())

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
<<<<<<< HEAD
                core.CPUPlace(),
                ["X", "Y"],
                "Out",
                user_defined_grads=[self.dx, self.dy],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)],
            )

        def matmul_grad(self, x, transpose_x, y, transpose_y):
            x = (
                np.transpose(x, self.shape_transpose_axes[x.ndim])
                if transpose_x
                else x
            )
            y = (
                np.transpose(y, self.shape_transpose_axes[y.ndim])
                if transpose_y
                else y
            )
=======
                core.CPUPlace(), ["X", "Y"],
                "Out",
                user_defined_grads=[self.dx, self.dy],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

        def matmul_grad(self, x, transpose_x, y, transpose_y):
            x = np.transpose(
                x, self.shape_transpose_axes[x.ndim]) if transpose_x else x
            y = np.transpose(
                y, self.shape_transpose_axes[y.ndim]) if transpose_y else y
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            return np.matmul(x, y)

        def calculate_grads(self):
            self.shape_transpose_axes = {
                2: [1, 0],
                3: [0, 2, 1],
                4: [0, 1, 3, 2],
                5: [0, 1, 2, 4, 3],
<<<<<<< HEAD
                6: [0, 1, 2, 3, 5, 4],
=======
                6: [0, 1, 2, 3, 5, 4]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

            # expand vector so it will be a valid matrix for multiplication
            if self.x_fp32.ndim == 1:
                self.x_fp32 = np.expand_dims(self.x_fp32, axis=0)
            if self.y_fp32.ndim == 1:
                self.y_fp32 = np.expand_dims(self.y_fp32, axis=1)

            x_transpose_axes = self.shape_transpose_axes[self.x_fp32.ndim]
            y_transpose_axes = self.shape_transpose_axes[self.y_fp32.ndim]

<<<<<<< HEAD
            x = (
                np.transpose(self.x_fp32, x_transpose_axes)
                if self.attrs['trans_x'] is True
                else self.x_fp32
            )
            y = (
                np.transpose(self.y_fp32, y_transpose_axes)
                if self.attrs['trans_y'] is True
                else self.y_fp32
            )
=======
            x = np.transpose(self.x_fp32, x_transpose_axes
                             ) if self.attrs['trans_x'] is True else self.x_fp32
            y = np.transpose(self.y_fp32, y_transpose_axes
                             ) if self.attrs['trans_y'] is True else self.y_fp32
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            dout = np.matmul(x, y)

            x_shape = x.shape
            y_shape = y.shape

            if x.ndim <= 2 or y.ndim <= 2:
                is_broadcast = False
            elif x.ndim != y.ndim:
                is_broadcast = True
            else:
                is_broadcast = x.shape[0:-2] != y.shape[0:-2]

            if self.attrs['trans_x'] is True and self.attrs['trans_y'] is True:
                self.dx = self.matmul_grad(self.y_fp32, True, dout, True)
                self.dy = self.matmul_grad(dout, True, self.x_fp32, True)
<<<<<<< HEAD
            elif (
                self.attrs['trans_x'] is True and self.attrs['trans_y'] is False
            ):
                self.dx = self.matmul_grad(self.y_fp32, False, dout, True)
                self.dy = self.matmul_grad(self.x_fp32, False, dout, False)
            elif (
                self.attrs['trans_x'] is False and self.attrs['trans_y'] is True
            ):
=======
            elif self.attrs['trans_x'] is True and self.attrs[
                    'trans_y'] is False:
                self.dx = self.matmul_grad(self.y_fp32, False, dout, True)
                self.dy = self.matmul_grad(self.x_fp32, False, dout, False)
            elif self.attrs['trans_x'] is False and self.attrs[
                    'trans_y'] is True:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                self.dx = self.matmul_grad(dout, False, self.y_fp32, False)
                self.dy = self.matmul_grad(dout, True, self.x_fp32, False)
            else:
                self.dx = self.matmul_grad(dout, False, self.y_fp32, True)
                self.dy = self.matmul_grad(self.x_fp32, True, dout, False)

            if is_broadcast:
                x_reduce_axis = []
                y_reduce_axis = []
                for index, (first, second) in enumerate(
<<<<<<< HEAD
                    zip(x_shape[0:-2], self.dx.shape[0:-2])
                ):
=======
                        zip(x_shape[0:-2], self.dx.shape[0:-2])):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    if first != second:
                        x_reduce_axis.append(index)

                for index, (first, second) in enumerate(
<<<<<<< HEAD
                    zip(y_shape[0:-2], self.dy.shape[0:-2])
                ):
=======
                        zip(y_shape[0:-2], self.dy.shape[0:-2])):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    if first != second:
                        y_reduce_axis.append(index)

                if x_reduce_axis:
<<<<<<< HEAD
                    self.dx = self.dx.sum(
                        axis=tuple(x_reduce_axis), keepdims=True
                    )
                if y_reduce_axis:
                    self.dy = self.dy.sum(
                        axis=tuple(y_reduce_axis), keepdims=True
                    )
=======
                    self.dx = self.dx.sum(axis=tuple(x_reduce_axis),
                                          keepdims=True)
                if y_reduce_axis:
                    self.dy = self.dy.sum(axis=tuple(y_reduce_axis),
                                          keepdims=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            # after multiplying with vector one dimension is deleted from tensor
            if len(x_shape) == 2 and x_shape[0] == 1:
                dout = dout.sum(axis=-2)
            if len(y_shape) == 2 and y_shape[1] == 1:
                dout = dout.sum(axis=-1)

            self.dout = dout

    cls_name = "{0}_{1}".format(parent.__name__, "BF16")
    TestMatMulV2Bf16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestMatMulV2Bf16OneDNNOp


create_bf16_test_class(TestMatMulV2VectorXMatrixTransposeYOneDNNOp)
create_bf16_test_class(TestMatMulV2VectorXMatrixOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXVectorTransposeXOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXVectorOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTransposeYOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix2OneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix3OneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTranposeXOneDNNOp2)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTranposeX2OneDNNOp3)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTransposeX3OneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix4OneDNNOp)
create_bf16_test_class(TestMatMulV2VectorXMatrix5DOneDNNOp)
create_bf16_test_class(TestMatMulV2Matrix3DXVectorOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTransposeXTransposeYOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrixTransposeY2OneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix5DTranposeYOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix6Dx2DOneDNNOp)
create_bf16_test_class(TestMatMulV2MatrixXMatrix2Dx5DOneDNNOp)


class TestMatMulV2OpTransposeReshapeEmptyFloat(
<<<<<<< HEAD
    TestMatMulOpTransposeReshapeEmptyFloat
):
=======
        TestMatMulOpTransposeReshapeEmptyFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type(self):
        self.op_type = "matmul_v2"


class TestMatMulV2OpTransposeReshapeBasicFloat(
<<<<<<< HEAD
    TestMatMulOpTransposeReshapeBasicFloat
):
=======
        TestMatMulOpTransposeReshapeBasicFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type(self):
        self.op_type = "matmul_v2"


class TestMatMulV2OpTransposeReshapeOtherDimFloat(
<<<<<<< HEAD
    TestMatMulOpTransposeReshapeOtherDimFloat
):
=======
        TestMatMulOpTransposeReshapeOtherDimFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type(self):
        self.op_type = "matmul_v2"


class TestMatMulV2OpReshapeTranspose(TestReshapeTransposeMatMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul_v2"
        self.transpose_y_name = "trans_y"


class TestMatMulV2OpReshapeTranspose4DXFloat(
<<<<<<< HEAD
    TestReshapeTransposeMatMulOp4DXFloat
):
=======
        TestReshapeTransposeMatMulOp4DXFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul_v2"
        self.transpose_y_name = "trans_y"


class TestMatMulV2OpReshapeTranspose4DYFloat(
<<<<<<< HEAD
    TestReshapeTransposeMatMulOp4DYFloat
):
=======
        TestReshapeTransposeMatMulOp4DYFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul_v2"
        self.transpose_y_name = "trans_y"


class TestMatMulV2OpReshapeTranspose4DXYFloat(
<<<<<<< HEAD
    TestReshapeTransposeMatMulOp4DXYFloat
):
=======
        TestReshapeTransposeMatMulOp4DXYFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul_v2"
        self.transpose_y_name = "trans_y"


class TestMatMulV2OpReshapeTranspose2DXFloat(
<<<<<<< HEAD
    TestReshapeTransposeMatMulOp2DXFloat
):
=======
        TestReshapeTransposeMatMulOp2DXFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul_v2"
        self.transpose_y_name = "trans_y"


class TestMatMulV2OpReshapeTranspose2DYFloat(
<<<<<<< HEAD
    TestReshapeTransposeMatMulOp2DYFloat
):
=======
        TestReshapeTransposeMatMulOp2DYFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul_v2"
        self.transpose_y_name = "trans_y"


class TestMatMulV2OpReshapeTranspose3DXFloat(
<<<<<<< HEAD
    TestReshapeTransposeMatMulOp3DXFloat
):
=======
        TestReshapeTransposeMatMulOp3DXFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul_v2"
        self.transpose_y_name = "trans_y"


class TestMatMulV2OpReshapeTranspose3DYFloat(
<<<<<<< HEAD
    TestReshapeTransposeMatMulOp3DYFloat
):
=======
        TestReshapeTransposeMatMulOp3DYFloat):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_type_and_transpose_y_name(self):
        self.op_type = "matmul_v2"
        self.transpose_y_name = "trans_y"


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
