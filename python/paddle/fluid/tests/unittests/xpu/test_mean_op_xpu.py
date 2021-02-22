#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

np.random.seed(10)


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out')


class TestMeanOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of mean_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, fluid.layers.mean, input1)
            # The input dtype of mean_op must be float16, float32, float64.
            input2 = fluid.layers.data(
                name='input2', shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.mean, input2)
            input3 = fluid.layers.data(
                name='input3', shape=[4], dtype="float16")
            fluid.layers.softmax(input3)


class TestXPUMeanOp(TestMeanOp):
    def init_dtype_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_checkout_grad(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
