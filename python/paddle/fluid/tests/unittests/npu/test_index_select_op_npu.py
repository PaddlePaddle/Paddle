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

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
from paddle.static import Program, program_guard

paddle.enable_static()
SEED = 2021


class TestNPUIndexSelect(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "index_select"
        self.config()

        x_np = np.random.random(self.x_shape).astype(self.x_type)
        index_np = np.random.randint(
            low=0, high=self.x_shape[self.dim], size=self.index_size)

        # compute real output as baseline.
        outer_loop = np.prod(self.x_shape[:self.dim])
        outer_loop = outer_loop.astype(self.index_type)
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

        self.inputs = {'X': x_np, 'Index': index_np}
        self.attrs = {'dim': self.dim}
        self.outputs = {'Out': out}

    # todo: comment second line when index_select grad npu op is ready. 
    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    # todo: replace first line with second line when index_select grad npu op is ready. 
    def test_check_grad(self):
        pass
        #self.check_grad_with_place(self.place, ['X'], 'Out')

    def config(self):
        self.x_shape = (100, 4, 5)
        self.x_type = np.float32
        self.dim = 1
        self.index_size = 100
        self.index_type = np.int64


class TestNPUIndexSelectCase2(TestNPUIndexSelect):
    def config(self):
        self.dim = -2
        self.x_type = np.float32
        self.index_type = np.int32
        self.x_shape = (10, 10, 4, 10)
        self.index_size = 10


class TestNPUIndexSelectAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0]]).astype('float32')
        self.data_index = np.array([0, 1, 1]).astype('int32')

    def test_index_select_api(self):
        paddle.set_device("npu:0")
        paddle.enable_static()
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            index = paddle.static.data(name='index', shape=[3], dtype='int32')
            z = paddle.index_select(x, index, axis=1)
            exe = paddle.static.Executor(paddle.NPUPlace(0))
            res, = exe.run(feed={'x': self.data_x,
                                 'index': self.data_index},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[1.0, 2.0, 2.0], [5.0, 6.0, 6.0],
                               [9.0, 10.0, 10.0]]).astype('float32')
        self.assertTrue(np.allclose(expect_out, np.array(res)))

        # case 2:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            index = paddle.static.data(name='index', shape=[3], dtype='int32')
            z = paddle.index_select(x, index)
            exe = paddle.static.Executor(paddle.NPUPlace(0))
            res, = exe.run(feed={'x': self.data_x,
                                 'index': self.data_index},
                           fetch_list=[z.name],
                           return_numpy=False)
        expect_out = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                               [5.0, 6.0, 7.0, 8.0]]).astype('float32')
        self.assertTrue(np.allclose(expect_out, np.array(res)))

    def test_dygraph_index_select_api(self):
        paddle.set_device("npu:0")
        paddle.disable_static()
        self.input_data()

        # case 1:
        x = paddle.to_tensor(self.data_x)
        index = paddle.to_tensor(self.data_index)
        z = paddle.index_select(x, index)
        np_z = z.numpy()
        expect_out = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                               [5.0, 6.0, 7.0, 8.0]]).astype('float32')
        self.assertTrue(np.allclose(expect_out, np_z))

        # case 2:
        x = paddle.to_tensor(self.data_x)
        index = paddle.to_tensor(self.data_index)
        z = paddle.index_select(x, index, axis=1)
        np_z = z.numpy()
        expect_out = np.array([[1.0, 2.0, 2.0], [5.0, 6.0, 6.0],
                               [9.0, 10.0, 10.0]]).astype('float32')
        self.assertTrue(np.allclose(expect_out, np_z))


if __name__ == '__main__':
    unittest.main()
