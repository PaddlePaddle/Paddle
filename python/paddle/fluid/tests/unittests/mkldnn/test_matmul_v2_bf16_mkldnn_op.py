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

from paddle.fluid.tests.unittests.op_test import convert_float_to_uint16
from paddle.fluid.tests.unittests.mkldnn.test_matmul_v2_mkldnn_op import TestMatMulV2VectorXVectorOneDNNOp, reference_matmul
import paddle.fluid.core as core
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
@unittest.skipIf(core.is_compiled_with_cuda(),
                 "core is compiled with CUDA which has no BF implementation")
class TestMatmulV2BF16VectorXVectorOneDNNOp(TestMatMulV2VectorXVectorOneDNNOp):
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
        self.x_shape = (3, 1, 5, 10)
        self.y_shape = (1, 2, 5, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatmulV2BF16MatrixXMatrix5DTranposeYOneDNNOp(
        TestMatmulV2BF16VectorXVectorOneDNNOp):
    def config(self):
        self.x_shape = (1, 3, 1, 4, 7)
        self.y_shape = (2, 1, 2, 2, 7)
        self.trans_x = False
        self.trans_y = True


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
