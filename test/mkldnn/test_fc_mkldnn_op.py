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


def fully_connected_naive(input, weights, bias_data):
    result = np.dot(input, weights) + bias_data
    return result


class MatrixGenerate:
    def __init__(self, mb, ic, oc, h, w):
        self.input = np.random.random((mb, ic * h * w)).astype("float32")
        self.weights = np.random.random((ic * h * w, oc)).astype("float32")


class TestFCMKLDNNOp(OpTest):
    def create_data(self):
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3)
        self.bias = np.random.random(15).astype("float32")

    def setUp(self):
        self.op_type = "fc"
        self._cpu_only = True
        self.use_mkldnn = True
        self.create_data()
        self.inputs = {
            'Input': self.matrix.input,
            'W': self.matrix.weights,
            'Bias': self.bias,
        }

        self.attrs = {'use_mkldnn': self.use_mkldnn}

        self.outputs = {
            'Out': fully_connected_naive(
                self.matrix.input, self.matrix.weights, self.bias
            )
        }

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


class TestFCMKLDNNOp1(TestFCMKLDNNOp):
    def create_data(self):
        self.matrix = MatrixGenerate(2, 15, 48, 2, 2)
        self.bias = np.random.random(48).astype("float32")


if __name__ == "__main__":
    import paddle

    paddle.enable_static()
    unittest.main()
