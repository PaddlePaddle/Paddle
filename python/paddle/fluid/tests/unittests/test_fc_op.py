# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest


def fc_refer(matrix, with_bias):
    in_n, in_c, in_h, in_w = matrix.input.shape
    w_i, w_o = matrix.weights.shape

    x_data = np.reshape(matrix.input, [in_n, in_c * in_h * in_w])
    w_data = np.reshape(matrix.weights, [w_i, w_o])
    b_data = np.reshape(matrix.bias, [1, w_o])
    result = None

    if with_bias:
        result = np.dot(x_data, w_data) + b_data
    else:
        result = np.dot(x_data, w_data)

    return result


class MatrixGenerate:
    def __init__(self, mb, ic, oc, h, w):
        self.input = np.random.random((mb, ic, h, w)).astype("float32")
        self.weights = np.random.random((ic * h * w, oc)).astype("float32")
        self.bias = np.random.random((1, oc)).astype("float32")


class TestFCOp(OpTest):
    def setUp(self):
        self.op_type = "fc"
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3)

        self.with_bias = True
        if self.with_bias:
            self.inputs = {
                'Input': self.matrix.input,
                'W': self.matrix.weights,
                'Bias': self.matrix.bias
            }
        else:
            self.inputs = {'Input': self.matrix.input, 'W': self.matrix.weights}

        self.attrs = {'use_mkldnn': False}

        self.outputs = {'Out': fc_refer(self.matrix, self.with_bias)}

    def test_check_output(self):
        self.check_output()


class TestFCOpNoBias(TestFCOp):
    def init_shapes(self, mb, ic, oc, h, w):
        self.with_bias = False
        self.matrix = MatrixGenerate(mb, ic, oc, h, w)


class TestFCOpWithBias(TestFCOp):
    def init_shapes(self, mb, ic, oc, h, w):
        self.with_bias = True
        self.matrix = MatrixGenerate(mb, ic, oc, h, w)


class TestFCOp1(TestFCOpNoBias):
    def init_op_type(self):
        self.init_shapes(2, 8, 10, 1, 1)


class TestFCOp2(TestFCOpNoBias):
    def init_op_type(self):
        self.init_shapes(4, 5, 6, 2, 2)


class TestFCOp4(TestFCOpNoBias):
    def init_op_type(self):
        self.init_shapes(1, 32, 64, 3, 3)


class TestFCOpWithBias1(TestFCOpWithBias):
    def init_op_type(self):
        self.init_shapes(3, 8, 10, 2, 1)


class TestFCOpWithBias2(TestFCOpWithBias):
    def init_op_type(self):
        self.init_shapes(4, 5, 6, 2, 2)


class TestFCOpWithBias3(TestFCOpWithBias):
    def init_op_type(self):
        self.init_shapes(1, 64, 32, 3, 3)


if __name__ == "__main__":
    unittest.main()
