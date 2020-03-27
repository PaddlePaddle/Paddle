#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle.fluid as fluid
import paddle.tensor as tensor
from paddle.fluid import Program, program_guard


class TestBmmOp(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        self.dtype = np.float32
        X = np.random.random((10, 3, 4)).astype("float32")
        Y = np.random.random((10, 4, 5)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class TeseBmmOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            input1_X = fluid.layers.data(
                name='input1_X', shape=[10, 3, 4, 2], dtype="float32")
            input1_Y = fluid.layers.data(
                name='input1_Y', shape=[10, 4, 5], dtype="float32")
            self.assertRaises(TypeError, tensor.bmm, input1_X, input1_Y)

            input2_X = fluid.layers.data(
                name='input2_X', shape=[10, 3, 4], dtype="float32")
            input2_Y = fluid.layers.data(
                name='input2_Y', shape=[10, 4, 5, 5], dtype="float32")
            self.assertRaises(TypeError, tensor.bmm, input2_X, input2_Y)

            input3_X = fluid.layers.data(
                name='input3_X', shape=[9, 3, 4], dtype="float32")
            input3_Y = fluid.layers.data(
                name='input3_Y', shape=[10, 4, 5], dtype="float32")
            self.assertRaises(TypeError, tensor.bmm, input3_X, input3_Y)

            input4_X = fluid.layers.data(
                name='input4_X', shape=[10, 3, 3], dtype="float32")
            input4_Y = fluid.layers.data(
                name='input4_Y', shape=[10, 4, 5], dtype="float32")
            self.assertRaises(TypeError, tensor.bmm, input4_X, input4_Y)


if __name__ == "__main__":
    unittest.main()
