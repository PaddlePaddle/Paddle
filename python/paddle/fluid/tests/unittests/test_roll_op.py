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
import paddle
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestRollOp(OpTest):
    def setUp(self):
        self.op_type = "roll"
        self.init_dtype_type()
        self.inputs = {'X': np.random.random(self.x_shape).astype(self.dtype)}
        self.attrs = {'shifts': self.shifts, 'dims': self.dims}
        self.outputs = {
            'Out': np.roll(self.inputs['X'], self.attrs['shifts'],
                           self.attrs['dims'])
        }

    def init_dtype_type(self):
        self.dtype = np.float64
        self.x_shape = (100, 4, 5)
        self.shifts = [101, -1]
        self.dims = [0, -2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestRollOpCase2(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float32
        self.x_shape = (100, 100, 5)
        self.shifts = [8, -1]
        self.dims = [-1, -2]


class TestRollAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    def test_index_select_api(self):
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 3])
            z = paddle.roll(x, shifts=1)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': self.data_x},
                           fetch_list=[z.name],
                           return_numpy=False)
            expect_out = np.array([[9.0, 1.0, 2.0], [3.0, 4.0, 5.0],
                                   [6.0, 7.0, 8.0]])
            self.assertTrue(np.allclose(expect_out, np.array(res)))

        # case 2:
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 3])
            z = paddle.roll(x, shifts=1, dims=0)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': self.data_x},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
        self.assertTrue(np.allclose(expect_out, np.array(res)))

    def test_dygraph_api(self):
        self.input_data()
        # case 1:
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            z = paddle.roll(x, shifts=1)
            np_z = z.numpy()
        expect_out = np.array([[9.0, 1.0, 2.0], [3.0, 4.0, 5.0],
                               [6.0, 7.0, 8.0]])
        self.assertTrue(np.allclose(expect_out, np_z))

        # case 2:
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            z = paddle.roll(x, shifts=1, dims=0)
            np_z = z.numpy()
        expect_out = np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
        self.assertTrue(np.allclose(expect_out, np_z))


if __name__ == "__main__":
    unittest.main()
