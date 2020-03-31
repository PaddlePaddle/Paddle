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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.tensor as tensor
import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.op import Operator


class TestWhere(OpTest):
    def setUp(self):
        self.op_type = "where"
        self.init_config()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X', 'Y'], 'Out')

    def init_config(self):
        self.x = np.random.uniform(-3, 5, (100)).astype("float64")
        self.x[np.abs(self.x) < 0.005] = 0.02
        self.y = np.zeros((100)).astype("float64")
        self.cond = np.zeros((100)).astype("bool")

        for i in range(0, len(self.x)):
            if self.x[i] > self.y[i]:
                self.cond[i] = True

        self.inputs = {'Condition': self.cond, 'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.where(self.cond, self.x, self.y)}


class TestWhereAPI(unittest.TestCase):
    def test_api(self):
        x = fluid.layers.data(name='x', shape=[4], dtype='float32')
        y = fluid.layers.data(name='y', shape=[4], dtype='float32')
        x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float32")
        y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype("float32")
        cond_i = np.array([False, False, True, True]).astype("bool")
        result = tensor.where(x > 1, X=x, Y=y)

        exe = fluid.Executor(fluid.CPUPlace())
        out = exe.run(fluid.default_main_program(),
                      feed={'x': x_i,
                            'y': y_i},
                      fetch_list=[result])
        assert np.array_equal(out[0], np.where(cond_i, x_i, y_i))


if __name__ == "__main__":
    unittest.main()
