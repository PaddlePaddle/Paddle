# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

class TestStack2DOneDNNOp(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (15, 7)
        self.axis = 0
        self.dtype = np.float32

    def initParameters(self):
        pass

    def getInputNames(self):
        input_names = []
        for i in range(self.num_inputs):
            input_names.append('x{}'.format(i))
        return input_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'stack'
        self.op_inputs = []

        for i in range(self.num_inputs):
            self.op_inputs.append(
                np.random.random(size=self.input_dim).astype(np.float32))

        input_list = []
        input_names = self.getInputNames()
        for i in range(self.num_inputs):
            input_list.append((input_names[i], self.op_inputs[i]))


        self.inputs = {'X': input_list}
        self.outputs = {'Y': np.stack(self.op_inputs, axis=self.axis)}
        self.attrs = {'axis': self.axis, 'use_mkldnn': True}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    # JUST FOR CI TO PASS, GRAD IS NOT IMPLEMENTED YET
    def test_check_grad(self):
        self.check_grad(['x0'], 'Y')


class TestStack3DOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = (6, 7, 8)
        self.num_inputs = 5
        self.axis = 1


class TestStack4DOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = (2, 4, 6, 8)
        self.num_inputs = 3
        self.axis = 4


class TestStack5DOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = (2, 3, 4, 5, 6)
        self.num_inputs = 6
        self.axis = 0


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
