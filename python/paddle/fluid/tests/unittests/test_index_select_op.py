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

import unittest
import paddle
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestIndexSelectOp(OpTest):

    def setUp(self):
        self.python_api = paddle.index_select
        self.op_type = "index_select"
        self.init_dtype_type()
        index_np = np.random.randint(low=0,
                                     high=self.x_shape[self.dim],
                                     size=self.index_size)
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {'X': x_np, 'Index': index_np}
        self.attrs = {'dim': self.dim}
        outer_loop = np.prod(self.x_shape[:self.dim])
        x_reshape = [outer_loop] + list(self.x_shape[self.dim:])
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))
        out_list = []
        for i in range(outer_loop):
            for j in range(self.index_size):
                out_list.append(x_np_reshape[i, index_np[j]])
        self.out_shape = list(self.x_shape)
        self.out_shape[self.dim] = self.index_size
        self.out_shape = tuple(self.out_shape)

        out = np.reshape(out_list, self.out_shape)
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.dim = 1
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (100, 4, 5)
        self.index_size = 100

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class TestIndexSelectOpCase2(TestIndexSelectOp):

    def init_dtype_type(self):
        self.x_type = np.float32
        self.index_type = np.int32
        self.dim = -2
        self.x_shape = (10, 10, 4, 10)
        self.index_size = 10


class TestIndexSelectOpCaseSingleThread(TestIndexSelectOp):

    def init_dtype_type(self):
        if fluid.is_compiled_with_cuda():
            fluid.set_flags({'FLAGS_cudnn_deterministic': True})
        self.x_type = np.float32
        self.index_type = np.int32
        self.dim = -2
        self.x_shape = (10, 10, 4, 10)
        self.index_size = 10


class TestIndexSelectAPI(unittest.TestCase):

    def input_data(self):
        self.data_x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0]])
        self.data_index = np.array([0, 1, 1]).astype('int32')

    def test_index_select_api(self):
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 4])
            index = fluid.layers.data(name='index',
                                      shape=[3],
                                      dtype='int32',
                                      append_batch_size=False)
            z = paddle.index_select(x, index, axis=1)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={
                'x': self.data_x,
                'index': self.data_index
            },
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[1.0, 2.0, 2.0], [5.0, 6.0, 6.0],
                               [9.0, 10.0, 10.0]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 2:
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[-1, 4])
            index = fluid.layers.data(name='index',
                                      shape=[3],
                                      dtype='int32',
                                      append_batch_size=False)
            z = paddle.index_select(x, index)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={
                'x': self.data_x,
                'index': self.data_index
            },
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                               [5.0, 6.0, 7.0, 8.0]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        self.input_data()
        # case 1:
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            index = fluid.dygraph.to_variable(self.data_index)
            z = paddle.index_select(x, index)
            np_z = z.numpy()
        expect_out = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                               [5.0, 6.0, 7.0, 8.0]])
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 2:
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            index = fluid.dygraph.to_variable(self.data_index)
            z = paddle.index_select(x, index, axis=1)
            np_z = z.numpy()
        expect_out = np.array([[1.0, 2.0, 2.0], [5.0, 6.0, 6.0],
                               [9.0, 10.0, 10.0]])
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
