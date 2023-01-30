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

<<<<<<< HEAD
import unittest

import numpy as np

=======
from __future__ import print_function

import unittest
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.fluid.tests.unittests.op_test import OpTest


def fully_connected_naive(input, weights, bias_data):
    result = np.dot(input, weights) + bias_data
    return result


class MatrixGenerate:
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, mb, ic, oc, h, w):
        self.input = np.random.random((mb, ic * h * w)).astype("float32")
        self.weights = np.random.random((ic * h * w, oc)).astype("float32")


class TestFCMKLDNNOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
            'Bias': self.bias,
=======
            'Bias': self.bias
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {'use_mkldnn': self.use_mkldnn}

        self.outputs = {
<<<<<<< HEAD
            'Out': fully_connected_naive(
                self.matrix.input, self.matrix.weights, self.bias
            )
=======
            'Out':
            fully_connected_naive(self.matrix.input, self.matrix.weights,
                                  self.bias)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


class TestFCMKLDNNOp1(TestFCMKLDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def create_data(self):
        self.matrix = MatrixGenerate(2, 15, 48, 2, 2)
        self.bias = np.random.random(48).astype("float32")


if __name__ == "__main__":
<<<<<<< HEAD
    import paddle

    paddle.enable_static()
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    unittest.main()
