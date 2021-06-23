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

from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
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

    Out = np.atleast_1d(np.matmul(X, Y))
    return Out


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
@unittest.skipIf(core.is_compiled_with_cuda(),
                 "core is compiled with CUDA which has no BF implementation")
class TestMatmulV2BF16VectorXVectorOneDNNOp(OpTest):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (100, )
        self.trans_x = False
        self.trans_y = False

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

        x_bf16 = convert_float_to_uint16(x)
        y_bf16 = convert_float_to_uint16(y)
        self.inputs = {
            'X': x_bf16,
            'Y': y_bf16,
        }

        self.attrs = {
            'trans_x': self.trans_x,
            'trans_y': self.trans_y,
            'use_mkldnn': True,
            'mkldnn_data_type': "bfloat16"
        }
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())


class TestMatmulV2BF16VectorXMatrixTransposeYOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatmulV2BF16VectorXMatrixOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100, )
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatmulV2BF16MatrixXVectorTransposeXOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100, )
        self.trans_x = True
        self.trans_y = False


class TestMatmulV2BF16MatrixXVectorTransposeX2OneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 2, 102, 1)
        self.y_shape = (102, )
        self.trans_x = True
        self.trans_y = False


class TestMatmulV2BF16MatrixXVectorOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100, )
        self.trans_x = False
        self.trans_y = False


class TestMatmulV2BF16MatrixXMatrixOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatmulV2BF16MatrixXMatrixTransposeYOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 1, 100)
        self.y_shape = (2, 1, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatmulV2BF16MatrixXMatrix2OneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 1, 25, 4)
        self.y_shape = (1, 2, 4, 25)
        self.trans_x = False
        self.trans_y = False


class TestMatmulV2BF16MatrixXMatrix3OneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatmulV2BF16MatrixXMatrixTranposeXOneDNNOp2(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 4, 25)
        self.y_shape = (1, 1, 4, 25)
        self.trans_x = True
        self.trans_y = False


class TestMatmulV2BF16MatrixXMatrixTranposeX2OneDNNOp3(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 2, 10, 10)
        self.y_shape = (2, 2, 10, 10)
        self.trans_x = True
        self.trans_y = False


class TestMatmulV2BF16MatrixXMatrixTransposeX3OneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = True
        self.trans_y = False


class TestMatmulV2BF16MatrixXMatrix4OneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = False
        self.trans_y = False


class TestMatmulV2BF16VectorXMatrix5DOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (100)
        self.y_shape = (1, 2, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatmulV2BF16Matrix3DXVectorOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = (100)
        self.trans_x = False
        self.trans_y = False


class TestMatmulV2BF16MatrixXMatrixTransposeXTransposeYOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = True
        self.trans_y = True


class TestMatmulV2BF16MatrixXMatrixTransposeY2OneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatmulV2BF16MatrixXMatrix5DTranposeYOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 3, 1, 10, 10)
        self.y_shape = (15, 1, 2, 10, 10)
        self.trans_x = False
        self.trans_y = True


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
