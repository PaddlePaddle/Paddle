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
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.framework import Operator


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
            'Bias': self.bias
        }

        self.attrs = {'use_mkldnn': self.use_mkldnn}

        self.outputs = {
            'Out': fully_connected_naive(self.matrix.input, self.matrix.weights,
                                         self.bias)
        }

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


# class TestMKLDNNTensorDumpOp(unittest.TestCase):
#     def setUp(self):
#         self.op_type = "fc"
#         self.init_data_type()
#         self.use_mkldnn = True
#         self.x = np.random.random((1, 10 * 3 * 3)).astype(self.dtype)
#         self.w = np.random.random((10 * 3 * 3, 15)).astype(self.dtype)
#         self.b = np.random.random(15).astype("float32")

#     def init_data_type(self):
#         self.dtype = np.float32

#     def test_check_output(self):
#         place = fluid.core.CPUPlace()
#         scope = fluid.core.Scope()
#         inputs = {'Input': ('x', self.x),
#                   'W':('w', self.w),
#                   'Bias':('b', self.b)}
#         outputs = {'Out':'dumpout'}

#         for input_key, input_value in inputs.items():
#             (var_name, var_value) = input_value
#             var = scope.var(var_name)
#             tensor = var.get_tensor()
#             tensor.set(var_value, place)

#         # fc_op = Operator(
#         #     type="fc", inputs={"Input":"x", "W":"w"}, outputs={"Out":"dumpout"})
#         # fc_op = Operator("fc", Input=["x"],W=["w"], Bias=["b"], Out=out_var_name)
# # block,
# #                  desc,
# #                  type=None,
# #                  inputs=None,
# #                  outputs=None,
# #                  attrs=None

#         fc_op.run(scope, place)
# out = scope.find_var("dumpout").get_tensor()
# out_array = np.array(out)
# expected_out = fully_connected_naive(self.x, self.w, self.b)
# self.assertTrue(
#     np.allclose(
#         expected_out, out_array, atol=1e-5),
#     "Inplace sum_mkldnn_op output has diff with expected output")
# print("WARNING! The Test run well")

if __name__ == "__main__":
    unittest.main()
