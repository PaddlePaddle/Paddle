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


class TestMVOp(OpTest):
    def setUp(self):
        self.op_type = "mv"
        self.init_config()
        self.inputs = {'X': self.x, 'Y': self.y}
        # tmpe_out = np.matmul(self.x, self.y)
        # self.outputs = {'Out': tmpe_out.reshape(tmpe_out.shape[0])}
        # tmpe_out = np.matmul(self.x, self.y)
        self.outputs = {'Out': np.matmul(self.x, self.y)}

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=1e-3)

    # def test_check_grad_ignore_x(self):
    #     self.check_grad(
    #         ['Y'], 'Out', max_relative_error=1e-3, no_grad_set=set("X"))

    # def test_check_grad_ignore_y(self):
    #     self.check_grad(
    #         ['X'], 'Out', max_relative_error=1e-3, no_grad_set=set('Y'))

    def init_config(self):
        # self.x = np.random.uniform(-1, 1, (5, 100)).astype("float32")
        # self.y = np.random.uniform(-1, 1, (100, 1)).astype("float32")
        self.x = np.random.random((5, 100)).astype("float64")
        self.y = np.random.random((100, 1)).astype("float64")


class Test_API_MV(unittest.TestCase):
    def test_dygraph_with_cpu(self):
        device = fluid.CPUPlace()
        with fluid.dygraph.guard(device):
            input_array1 = np.random.rand(30, 400).astype("float64")
            input_array2 = np.random.rand(400, 1).astype("float64")
            data1 = fluid.dygraph.to_variable(input_array1)
            data2 = fluid.dygraph.to_variable(input_array2)
            out = paddle.mv(data1, data2)
            # print('-----out-----: ', out)
            expected_result = np.matmul(input_array1, input_array2)
            expected_result.reshape(expected_result.shape[0])
        self.assertTrue(np.allclose(expected_result, out.numpy()))

    def test_dygraph_with_gpu(self):
        device = fluid.CUDAPlace(0)
        with fluid.dygraph.guard(device):
            input_array1 = np.random.rand(30, 400).astype("float64")
            input_array2 = np.random.rand(400, 1).astype("float64")
            data1 = fluid.dygraph.to_variable(input_array1)
            data2 = fluid.dygraph.to_variable(input_array2)
            out = paddle.mv(data1, data2)
            expected_result = np.matmul(input_array1, input_array2)
            expected_result.reshape(expected_result.shape[0])
        self.assertTrue(np.allclose(expected_result, out.numpy()))


if __name__ == '__main__':
    unittest.main()
