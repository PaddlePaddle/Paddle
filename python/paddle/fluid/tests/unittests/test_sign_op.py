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
from paddle.fluid import Program, program_guard


class TestSignOp(OpTest):
    def setUp(self):
        self.op_type = "sign"
        self.inputs = {
            'X': np.random.uniform(-10, 10, (10, 10)).astype("float32")
        }
        self.outputs = {'Out': np.sign(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSignOpError(OpTest):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of sign_op must be Variable or numpy.ndarray.
            input1 = 12
            self.assertRaises(TypeError, fluid.layers.sign, input1)
            # The input dtype of sign_op must be float32, float64.
            input2 = fluid.layers.data(
                name='input2', shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.sign, input2)


if __name__ == "__main__":
    unittest.main()
