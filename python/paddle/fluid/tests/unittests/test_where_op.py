#Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.op import Operator
from paddle.fluid.backward import append_backward


class TestWhereOp(OpTest):
    def setUp(self):
        self.op_type = "where"
        self.init_config()
        self.inputs = {'Condition': self.cond, 'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.where(self.cond, self.x, self.y)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def init_config(self):
        self.x = np.random.uniform(-3, 5, (100)).astype("float64")
        self.y = np.random.uniform(-3, 5, (100)).astype("float64")
        self.cond = np.zeros((100)).astype("bool")


class TestWhereOp2(TestWhereOp):
    def init_config(self):
        self.x = np.random.uniform(-5, 5, (60, 2)).astype("float64")
        self.y = np.random.uniform(-5, 5, (60, 2)).astype("float64")
        self.cond = np.ones((60, 2)).astype("bool")


class TestWhereOp3(TestWhereOp):
    def init_config(self):
        self.x = np.random.uniform(-3, 5, (20, 2, 4)).astype("float64")
        self.y = np.random.uniform(-3, 5, (20, 2, 4)).astype("float64")
        self.cond = np.array(np.random.randint(2, size=(20, 2, 4)), dtype=bool)


if __name__ == '__main__':
    unittest.main()
