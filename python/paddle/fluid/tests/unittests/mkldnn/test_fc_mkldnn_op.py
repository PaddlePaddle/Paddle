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
from numpy.core.numeric import True_
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool
import paddle


def fully_connected_naive(input, weights, bias_data):
    result = np.dot(input, weights)
    if bias_data is not None:
        result = result + bias_data

    return result


@OpTestTool.skip_if_not_cpu()
class TestFCOneDNNKernel2DNoBias(OpTest):
    def init_shape(self):
        self.mb = 12
        self.ic = 10
        self.oc = 15
        self.h = 3
        self.w = 3

    def init_rank_and_bias(self):
        self.input_rank = 2
        self.with_bias = None

    def generate_data(self):
        self.input = np.random.random(
            (self.mb, self.ic * self.h * self.w)).astype("float32")
        self.weights = np.random.random(
            (self.ic * self.h * self.w, self.oc)).astype("float32")
        self.np_weights = self.weights.copy()

        if self.with_bias is not None:
            self.bias = np.random.random((self.oc)).astype("float32")
        else:
            self.bias = None

        self.output = fully_connected_naive(self.input, self.np_weights,
                                            self.bias)

    def setUp(self):
        self.op_type = "fc"
        self.use_mkldnn = True
        self.init_shape()
        self.init_rank_and_bias()
        self.generate_data()

        if self.input_rank == 3:
            self.input = np.reshape(self.input, (self.mb // 2, 2,
                                                 self.ic * self.h * self.w))
            self.output = np.reshape(self.output, (self.mb // 2, 2, self.oc))
        elif self.input_rank == 4:
            self.input = np.reshape(self.input, (self.mb // 6, 3, 2,
                                                 self.ic * self.h * self.w))
            self.output = np.reshape(self.output, (self.mb // 6, 3, 2, self.oc))

        self.inputs = {'Input': self.input, 'W': self.weights}

        if self.bias is not None:
            self.inputs['Bias'] = self.bias

        self.attrs = {
            'use_mkldnn': self.use_mkldnn,
            'in_num_col_dims': self.input_rank - 1
        }

        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


class TestFCOneDNNKernel3DWithBias(TestFCOneDNNKernel2DNoBias):
    def init_rank_and_bias(self):
        self.input_rank = 3
        self.with_bias = True


class TestFCOneDNNKernel3DNoBias(TestFCOneDNNKernel2DNoBias):
    def init_rank_and_bias(self):
        self.input_rank = 3
        self.with_bias = False


class TestFCOneDNNKernel4DWithBias(TestFCOneDNNKernel2DNoBias):
    def init_rank_and_bias(self):
        self.input_rank = 4
        self.with_bias = True


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
