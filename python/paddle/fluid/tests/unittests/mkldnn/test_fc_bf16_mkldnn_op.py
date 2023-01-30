# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
from paddle import enable_static
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
=======
from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
from paddle import enable_static
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def fully_connected_naive(input, weights, bias_data):
    result = np.dot(input, weights) + bias_data
    return result


class MatrixGenerate:
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, mb, ic, oc, h, w):
        self.input = np.random.random((mb, ic * h * w)).astype(np.float32)
        self.weights = np.random.random((ic * h * w, oc)).astype(np.float32)


<<<<<<< HEAD
@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestFcBf16MklDNNOp(OpTest):
=======
@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestFcBf16MklDNNOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3)
        self.bias = np.random.random(15).astype("float32")

    def setUp(self):
        self.op_type = "fc"
        self.use_mkldnn = True
        self.mkldnn_data_type = "bfloat16"
        self.force_fp32_output = False
        self.generate_data()

<<<<<<< HEAD
        self.output = fully_connected_naive(
            self.matrix.input, self.matrix.weights, self.bias
        )
=======
        self.output = fully_connected_naive(self.matrix.input,
                                            self.matrix.weights, self.bias)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not self.force_fp32_output:
            self.output = convert_float_to_uint16(self.output)

        self.inputs = {
            'Input': convert_float_to_uint16(self.matrix.input),
            'W': self.matrix.weights,
<<<<<<< HEAD
            'Bias': self.bias,
=======
            'Bias': self.bias
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {
            'use_mkldnn': self.use_mkldnn,
<<<<<<< HEAD
            'force_fp32_output': self.force_fp32_output,
=======
            'force_fp32_output': self.force_fp32_output
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


class TestFCMKLDNNOp1(TestFcBf16MklDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generate_data(self):
        self.matrix = MatrixGenerate(2, 15, 48, 2, 2)
        self.bias = np.random.random(48).astype(np.float32)


if __name__ == "__main__":
    enable_static()
    unittest.main()
