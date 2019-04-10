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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest


def fully_connected_naive(input, weights, bias_data=None):
    in_n, in_c, in_h, in_w = input.shape
    w_h, w_c = weights.shape

    x_data = np.reshape(input, [in_n, in_c * in_h * in_w])
    # this transpose should be implemented at C code
    w_data = np.transpose(np.reshape(weights, (w_c, in_c * in_h * in_w)))
    result = None

    if not bias_data:
        result = np.dot(x_data, w_data)
    else:
        result = np.dot(x_data, w_data) + bias_data

    return result


class MatrixGenerate:
    def __init__(self, mb, ic, oc, h, w):
        self.input = np.random.random((mb, ic, h, w)).astype("float32")
        self.weights = np.random.random((ic * h * w, oc)).astype("float32")


class TestFCMKLDNNOp(OpTest):
    def setUp(self):
        self.op_type = "fc"
        self.use_mkldnn = True
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3)

        self.inputs = {'Input': self.matrix.input, 'W': self.matrix.weights}

        self.attrs = {'use_mkldnn': self.use_mkldnn, }

        self.outputs = {
            'Out': fully_connected_naive(self.matrix.input, self.matrix.weights)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(set(['Input', 'W']), 'Out', max_relative_error=0.9)

    def test_check_grad_no_weight(self):
        self.check_grad(
            ['Input'], 'Out', max_relative_error=0.5, no_grad_set=set('W'))


class TestFCMKLDNNOp1(TestFCMKLDNNOp):
    def init_op_type(self):
        self.matrix = MatrixGenerate(2, 15, 48, 2, 2)


class TestFCMKLDNNOp2(TestFCMKLDNNOp):
    def init_op_type(self):
        self.matrix = MatrixGenerate(2, 32, 40, 1, 1)


class TestFCMKLDNNOp3(TestFCMKLDNNOp):
    def init_op_type(self):
        self.matrix = MatrixGenerate(2, 2, 4, 1, 1)


class TestFCMKLDNNOp4(TestFCMKLDNNOp):
    def init_op_type(self):
        self.matrix = MatrixGenerate(2, 32, 48, 2, 2)


class TestFCMKLDNNOp4(TestFCMKLDNNOp):
    def init_op_type(self):
        self.matrix = MatrixGenerate(2, 32, 1000, 6, 6)


if __name__ == "__main__":
    unittest.main()
