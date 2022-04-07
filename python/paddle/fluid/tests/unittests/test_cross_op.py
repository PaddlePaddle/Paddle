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
import paddle.fluid.core as core
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestCrossOp(OpTest):
    def setUp(self):
        self.op_type = "cross"
        self.python_api = paddle.cross
        self.initTestCase()
        self.inputs = {
            'X': np.random.random(self.shape).astype(self.dtype),
            'Y': np.random.random(self.shape).astype(self.dtype)
        }
        self.init_output()

    def initTestCase(self):
        self.attrs = {'dim': -2}
        self.dtype = np.float64
        self.shape = (1024, 3, 1)

    def init_output(self):
        x = np.squeeze(self.inputs['X'], 2)
        y = np.squeeze(self.inputs['Y'], 2)
        z_list = []
        for i in range(1024):
            z_list.append(np.cross(x[i], y[i]))
        self.outputs = {'Out': np.array(z_list).reshape(self.shape)}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_eager=True)


class TestCrossOpCase1(TestCrossOp):
    def initTestCase(self):
        self.shape = (2048, 3)
        self.dtype = np.float32

    def init_output(self):
        z_list = []
        for i in range(2048):
            z_list.append(np.cross(self.inputs['X'][i], self.inputs['Y'][i]))
        self.outputs = {'Out': np.array(z_list).reshape(self.shape)}


class TestCrossAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        self.data_y = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_cross_api(self):
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 3])
            y = fluid.layers.data(name='y', shape=[-1, 3])
            z = paddle.cross(x, y, axis=1)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': self.data_x,
                                 'y': self.data_y},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(expect_out, np.array(res)))

        # case 2:
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 3])
            y = fluid.layers.data(name='y', shape=[-1, 3])
            z = paddle.cross(x, y)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': self.data_x,
                                 'y': self.data_y},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0],
                               [-1.0, -1.0, -1.0]])
        self.assertTrue(np.allclose(expect_out, np.array(res)))

        # case 3:
        with program_guard(Program(), Program()):
            x = fluid.data(name="x", shape=[-1, 3], dtype="float32")
            y = fluid.data(name='y', shape=[-1, 3], dtype='float32')

            y_1 = paddle.cross(x, y, name='result')
            self.assertEqual(('result' in y_1.name), True)

    def test_dygraph_api(self):
        self.input_data()
        # case 1:
        # with fluid.dygraph.guard():
        #     x = fluid.dygraph.to_variable(self.data_x)
        #     y = fluid.dygraph.to_variable(self.data_y)
        #     z = paddle.cross(x, y)
        #     np_z = z.numpy()
        # expect_out = np.array([[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0],
        #                        [-1.0, -1.0, -1.0]])
        # self.assertTrue(np.allclose(expect_out, np_z))

        # case 2:
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            y = fluid.dygraph.to_variable(self.data_y)
            z = paddle.cross(x, y, axis=1)
            np_z = z.numpy()
        expect_out = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(expect_out, np_z))


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
