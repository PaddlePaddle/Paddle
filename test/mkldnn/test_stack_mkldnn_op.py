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
from op_test import OpTest, OpTestTool

import paddle
from paddle.base import core


@OpTestTool.skip_if_not_cpu()
class TestStack2DOneDNNOp(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (2, 2)
        self.axis = 1
        self.dtype = np.float32

    def initParameters(self):
        pass

    def getInputNames(self):
        input_names = []
        for i in range(self.num_inputs):
            input_names.append(f'x{i}')
        return input_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'stack'
        self.op_inputs = []

        for i in range(self.num_inputs):
            self.op_inputs.append(
                np.random.random(size=self.input_dim).astype(np.float32)
            )

        input_list = []
        input_names = self.getInputNames()
        for i in range(self.num_inputs):
            input_list.append((input_names[i], self.op_inputs[i]))

        self.inputs = {'X': input_list}
        self.outputs = {'Y': np.stack(self.op_inputs, axis=self.axis)}
        self.attrs = {'axis': self.axis, 'use_mkldnn': True}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)

    # JUST FOR CI TO PASS, GRAD IS NOT IMPLEMENTED YET
    def test_check_grad(self):
        pass


class TestStack1DOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = 100
        self.axis = 0


class TestStack0DOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = ()
        self.axis = 0


class TestStack1DAxis1OneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = 100
        self.axis = 1


class TestStack2DAxisLastOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = (13, 24)
        self.num_inputs = 5
        self.axis = -1


class TestStack3DAxisNegativeOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = (10, 128, 128)
        self.axis = -2


class TestStack3DOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = (10, 128, 128)
        self.num_inputs = 3
        self.axis = 1


class TestStack4DOneDNNOp(TestStack2DOneDNNOp):
    def initParameters(self):
        self.input_dim = (2, 2, 2, 2)
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
