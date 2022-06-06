#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from termios import N_PPP  #   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid.core as core

from paddle.fluid import Program, program_guard, Executor, default_main_program


class TestPad3dOneDNNOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.value = 0.0
        self.initTestCase()
        self.op_type = "pad3d"
        self.python_api = paddle.nn.functional.pad
        self.inputs = {'X': np.random.random(self.shape).astype("float32")}
        self.attrs = {'use_mkldnn': True}
        if self.variable_paddings:
            self.attrs['paddings'] = []
            self.inputs['Paddings'] = np.array(
                self.paddings).flatten().astype("int32")
        else:
            self.attrs['paddings'] = np.array(
                self.paddings).flatten().astype("int32")
        self.attrs['value'] = self.value
        self.attrs['mode'] = self.mode
        self.attrs['data_format'] = self.data_format
        if self.data_format == "NCDHW":
            paddings = [
                (0, 0),
                (0, 0),
                (self.paddings[4], self.paddings[5]),
                (self.paddings[2], self.paddings[3]),
                (self.paddings[0], self.paddings[1]),
            ]
        else:
            paddings = [
                (0, 0),
                (self.paddings[4], self.paddings[5]),
                (self.paddings[2], self.paddings[3]),
                (self.paddings[0], self.paddings[1]),
                (0, 0),
            ]
        if self.mode == "constant":
            out = np.pad(self.inputs['X'],
                         paddings,
                         mode=self.mode,
                         constant_values=self.value)
        elif self.mode == "reflect":
            out = np.pad(self.inputs['X'], paddings, mode=self.mode)
        elif self.mode == "replicate":
            out = np.pad(self.inputs['X'], paddings, mode="edge")
        elif self.mode == "circular":
            out = np.pad(self.inputs['X'], paddings, mode="wrap")
        self.outputs = {'Out': out}

    def test_check_output(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        print(self.inputs["X"].shape)
        print(self.outputs["Out"].shape)
        # print("\n\n\n")
        # print("inputs", self.inputs["X"])
        # print("\n\n\n")
        # print("outputs", self.outputs["Out"])
        # print("\n\n\n")
        self.check_output()

    # def test_check_grad_normal(self):
    #     self.check_grad(['X'], 'Out')

    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 0, 0, 0, 0, 0]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.pad_value = 0.0
        self.variable_paddings = False


class TestCase1(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.value = 1.0
        self.variable_paddings = False



if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
