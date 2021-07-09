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

from __future__ import print_function

import unittest
import numpy as np

from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle.fluid.core as core
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework

paddle.enable_static()


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

    Out = np.atleast_1d(np.matmul(X, Y))
    return Out


class TestMatMulV2VectorXVectorOneDNNOp(OpTest):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (100, )
        self.trans_x = False
        self.trans_y = False

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
        result = reference_matmul(x, y, self.trans_x,
                                  self.trans_y).astype("float32")

        self.set_inputs(x, y)
        self.attrs = {
            'trans_x': self.trans_x,
            'trans_y': self.trans_y,
            'use_mkldnn': True
        }
        self.set_dtype_attr()
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestMatMulV2VectorXMatrixTransposeYOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2VectorXMatrixOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXVectorTransposeXOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100, )
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXVectorOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100, )
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 1)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeYOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 1, 100)
        self.y_shape = (2, 1, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2MatrixXMatrix2OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 12, 4)
        self.y_shape = (1, 2, 4, 12)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrix3OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTranposeXOneDNNOp2(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 4, 25)
        self.y_shape = (1, 1, 4, 25)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTranposeX2OneDNNOp3(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 2, 5, 4)
        self.y_shape = (2, 2, 5, 3)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeX3OneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 6, 5)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2MatrixXMatrix4OneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2VectorXMatrix5DOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100)
        self.y_shape = (1, 2, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2Matrix3DXVectorOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 40)
        self.y_shape = (40)
        self.trans_x = False
        self.trans_y = False


class TestMatMulV2MatrixXMatrixTransposeXTransposeYOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 10, 8)
        self.y_shape = (1, 2, 9, 10)
        self.trans_x = True
        self.trans_y = True


class TestMatMulV2MatrixXMatrixTransposeY2OneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 9, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2MatrixXMatrix5DTranposeYOneDNNOp(
        TestMatMulV2VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 3, 1, 10, 10)
        self.y_shape = (3, 1, 2, 9, 10)
        self.trans_x = False
        self.trans_y = True


#   BF16 TESTS
def create_bf16_test_class(parent):
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestMatMulV2Bf16OneDNNOp(parent):
        def set_inputs(self, x, y):
            self.inputs = {
                'X': convert_float_to_uint16(x),
                'Y': convert_float_to_uint16(y)
            }

        def set_dtype_attr(self):
            self.attrs['mkldnn_data_type'] = "bfloat16"

        def test_check_output(self):
            self.check_output_with_place(core.CPUPlace())

        def test_check_grad(self):
            pass

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

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
