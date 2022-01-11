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


class TestRepeatInterleaveOp(OpTest):
    def setUp(self):
        self.op_type = "repeat_interleave"
        self.init_dtype_type()
        index_np = np.random.randint(
            low=0, high=3, size=self.index_size).astype(self.index_type)
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {'X': x_np, 'RepeatsTensor': index_np}
        self.attrs = {'dim': self.dim}

        outer_loop = np.prod(self.x_shape[:self.dim])
        x_reshape = [outer_loop] + list(self.x_shape[self.dim:])
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))
        out_list = []
        for i in range(outer_loop):
            for j in range(self.index_size):
                for k in range(index_np[j]):
                    out_list.append(x_np_reshape[i, j])
        self.out_shape = list(self.x_shape)
        self.out_shape[self.dim] = np.sum(index_np)
        self.out_shape = tuple(self.out_shape)

        out = np.reshape(out_list, self.out_shape)
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.dim = 1
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (8, 4, 5)
        self.index_size = self.x_shape[self.dim]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestRepeatInterleaveOp2(OpTest):
    def setUp(self):
        self.op_type = "repeat_interleave"
        self.init_dtype_type()
        index_np = 2
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {'X': x_np}  #, 'RepeatsTensor': None}
        self.attrs = {'dim': self.dim, 'Repeats': index_np}

        outer_loop = np.prod(self.x_shape[:self.dim])
        x_reshape = [outer_loop] + list(self.x_shape[self.dim:])
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))
        out_list = []
        for i in range(outer_loop):
            for j in range(self.index_size):
                for k in range(index_np):
                    out_list.append(x_np_reshape[i, j])
        self.out_shape = list(self.x_shape)
        self.out_shape[self.dim] = index_np * self.index_size
        self.out_shape = tuple(self.out_shape)

        out = np.reshape(out_list, self.out_shape)
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.dim = 1
        self.x_type = np.float64
        self.x_shape = (8, 4, 5)
        self.index_size = self.x_shape[self.dim]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestIndexSelectAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0]])
        self.data_index = np.array([0, 1, 2, 1]).astype('int32')

    def test_repeat_interleave_api(self):
        paddle.enable_static()
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 4])
            index = fluid.layers.data(
                name='repeats',
                shape=[4],
                dtype='int32',
                append_batch_size=False)
            z = paddle.repeat_interleave(x, index, axis=1)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': self.data_x,
                                 'repeats': self.data_index},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.repeat(self.data_x, self.data_index, axis=1)
        self.assertTrue(np.allclose(expect_out, np.array(res)))

        # case 2:
        repeats = np.array([1, 2, 1]).astype('int32')
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 4])
            index = fluid.layers.data(
                name='repeats',
                shape=[3],
                dtype='int32',
                append_batch_size=False)
            z = paddle.repeat_interleave(x, index, axis=0)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={
                'x': self.data_x,
                'repeats': repeats,
            },
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.repeat(self.data_x, repeats, axis=0)
        self.assertTrue(np.allclose(expect_out, np.array(res)))

        repeats = 2
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 4])
            z = paddle.repeat_interleave(x, repeats, axis=0)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={'x': self.data_x},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.repeat(self.data_x, repeats, axis=0)
        self.assertTrue(np.allclose(expect_out, np.array(res)))

    def test_dygraph_api(self):
        self.input_data()
        # case axis none
        input_x = np.array([[1, 2, 1], [1, 2, 3]]).astype('int32')
        index_x = np.array([1, 1, 2, 1, 2, 2]).astype('int32')

        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(input_x)
            index = fluid.dygraph.to_variable(index_x)
            z = paddle.repeat_interleave(x, index, None)
            np_z = z.numpy()
        expect_out = np.repeat(input_x, index_x, axis=None)
        self.assertTrue(np.allclose(expect_out, np_z))

        # case repeats int
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(input_x)
            index = 2
            z = paddle.repeat_interleave(x, index, None)
            np_z = z.numpy()
        expect_out = np.repeat(input_x, index, axis=None)
        self.assertTrue(np.allclose(expect_out, np_z))

        # case 1:
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            index = fluid.dygraph.to_variable(self.data_index)
            z = paddle.repeat_interleave(x, index, -1)
            np_z = z.numpy()
        expect_out = np.repeat(self.data_x, self.data_index, axis=-1)
        self.assertTrue(np.allclose(expect_out, np_z))

        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            index = fluid.dygraph.to_variable(self.data_index)
            z = paddle.repeat_interleave(x, index, 1)
            np_z = z.numpy()
        expect_out = np.repeat(self.data_x, self.data_index, axis=1)
        self.assertTrue(np.allclose(expect_out, np_z))

        # case 2:
        index_x = np.array([1, 2, 1]).astype('int32')
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            index = fluid.dygraph.to_variable(index_x)
            z = paddle.repeat_interleave(x, index, axis=0)
            np_z = z.numpy()
        expect_out = np.repeat(self.data_x, index, axis=0)
        self.assertTrue(np.allclose(expect_out, np_z))


if __name__ == '__main__':
    unittest.main()
