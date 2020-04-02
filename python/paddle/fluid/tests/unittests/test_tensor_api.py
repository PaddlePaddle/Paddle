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
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestTensorAPI(unittest.TestCase):
    def test_nonzero_api(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[-1, 2])
            x_1_d = fluid.layers.data(name='x_1_d', shape=[-1])
            y_tuple = paddle.nonzero(x, as_tuple=True)
            y_1_tuple = paddle.nonzero(x_1_d, as_tuple=True)
            self.assertEqual(len(y_tuple), 2)
            self.assertEqual(len(y_1_tuple), 1)
            y = paddle.nonzero(x, as_tuple=False)
            y1 = paddle.nonzero(x_1_d, as_tuple=False)
            z = fluid.layers.concat(list(y_tuple), axis=1)
            z1 = fluid.layers.concat(list(y_1_tuple), axis=1)
        data = np.array([[True, False], [False, True]])
        data_1_d = np.array([True, True, False])
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        outs = exe.run(main_program,
                       feed={'x': data,
                             'x_1_d': data_1_d},
                       fetch_list=[z.name, z1.name],
                       return_numpy=False)
        expect_out = np.array([[0, 0], [1, 1]])
        expect_1d_out = np.array([[0], [1]])
        self.assertTrue(np.allclose(expect_out, np.array(outs[0])))
        self.assertTrue(np.allclose(expect_1d_out, np.array(outs[1])))

    def test_cross_api(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[-1, 3])
            y = fluid.layers.data(name='y', shape=[-1, 3])
            z_default = paddle.cross(x, y)
            z = paddle.cross(x, y, dim=1)
        data_x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        data_y = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        outs = exe.run(main_program,
                       feed={'x': data_x,
                             'y': data_y},
                       fetch_list=[z_default.name, z.name],
                       return_numpy=False)
        expect_default_out = np.array([[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0],
                                       [-1.0, -1.0, -1.0]])
        expect_out = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(expect_default_out, np.array(outs[0])))
        self.assertTrue(np.allclose(expect_out, np.array(outs[1])))

    def test_roll_api(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[-1, 3])
            z_default = paddle.roll(x, shifts=1)
            z = paddle.roll(x, shifts=1, dims=0)
        data_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        outs = exe.run(main_program,
                       feed={'x': data_x},
                       fetch_list=[z_default.name, z.name],
                       return_numpy=False)
        expect_default_out = np.array([[9.0, 1.0, 2.0], [3.0, 4.0, 5.0],
                                       [6.0, 7.0, 8.0]])
        expect_out = np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
        self.assertTrue(np.allclose(expect_default_out, np.array(outs[0])))
        self.assertTrue(np.allclose(expect_out, np.array(outs[1])))

    def test_index_select_api(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[-1, 4])
            index = fluid.layers.data(
                name='index', shape=[3], dtype='int32', append_batch_size=False)
            z_default = paddle.index_select(x, index)
            z = paddle.index_select(x, index, dim=1)
        data_x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0]])
        data_index = np.array([0, 1, 1]).astype('int32')
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        outs = exe.run(main_program,
                       feed={'x': data_x,
                             'index': data_index},
                       fetch_list=[z_default.name, z.name],
                       return_numpy=False)
        expect_default_out = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0]])
        expect_out = np.array([[1.0, 2.0, 2.0], [5.0, 6.0, 6.0],
                               [9.0, 10.0, 10.0]])
        self.assertTrue(np.allclose(expect_default_out, np.array(outs[0])))
        self.assertTrue(np.allclose(expect_out, np.array(outs[1])))


if __name__ == "__main__":
    unittest.main()
