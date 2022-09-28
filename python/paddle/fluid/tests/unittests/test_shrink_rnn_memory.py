#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.executor import Executor
import paddle.fluid.layers as layers
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import default_main_program, switch_main_program
from paddle.fluid.framework import Program, program_guard
import numpy as np

from paddle.fluid.layers.control_flow import shrink_memory
from paddle.fluid.layers.control_flow import lod_rank_table


class TestShrinkRNNMemoryBase(unittest.TestCase):

    def setUp(self):
        self.main_program = Program()
        switch_main_program(self.main_program)
        x = layers.data('x', shape=[100], dtype='float32')
        x.stop_gradient = False
        rank_table_tensor = layers.data('rank_table_tensor',
                                        shape=[1],
                                        dtype='float32',
                                        lod_level=1)
        table = lod_rank_table(x=rank_table_tensor)
        i = layers.zeros(dtype='int64', shape=[1])
        self.mem1 = shrink_memory(x=x, i=i, table=table)
        i = layers.increment(x=i)
        i.stop_gradient = True
        self.mem2 = shrink_memory(x=self.mem1, i=i, table=table)
        i = layers.increment(x=i)
        i.stop_gradient = True
        self.mem3 = shrink_memory(x=self.mem2, i=i, table=table)
        mem3_mean = paddle.mean(self.mem3)
        append_backward(loss=mem3_mean)
        self.x_grad = self.main_program.global_block().var('x@GRAD')

    def sum_lodtensor(self, tensor):
        sum_res = 0.0
        for i in range(np.product(tensor.shape())):
            sum_res += tensor._get_float_element(i)
        return sum_res


class TestShrinkRNNMemoryReferLoD(TestShrinkRNNMemoryBase):

    def test_refer_lod(self):
        cpu = core.CPUPlace()
        x_tensor = core.LoDTensor()
        x_tensor.set_recursive_sequence_lengths([[2, 3, 1]])
        tensor_np = np.random.random(size=(6, 100)).astype('float32')
        x_tensor.set(tensor_np, cpu)

        rank_table_tensor = core.LoDTensor()
        rank_table_tensor.set_recursive_sequence_lengths([[1, 2, 3]])
        rank_table_tensor.set(
            np.random.random(size=(6, 1)).astype('float32'), cpu)

        exe = Executor(cpu)
        outs = exe.run(
            feed={
                'x': x_tensor,
                'rank_table_tensor': rank_table_tensor
            },
            fetch_list=[self.mem1, self.mem2, self.mem3, self.x_grad],
            return_numpy=False)
        np.testing.assert_allclose(tensor_np[0:6], outs[0], rtol=1e-05)
        np.testing.assert_allclose(tensor_np[0:5], outs[1], rtol=1e-05)
        np.testing.assert_allclose(tensor_np[0:2], outs[2], rtol=1e-05)
        self.assertAlmostEqual(1.0, self.sum_lodtensor(outs[3]), delta=0.01)


class TestShrinkRNNMemoryNoLoD(TestShrinkRNNMemoryBase):

    def test_no_lod(self):
        cpu = core.CPUPlace()
        x_tensor = core.LoDTensor()
        tensor_np = np.random.random(size=(3, 100)).astype('float32')
        x_tensor.set(tensor_np, cpu)

        rank_table_tensor = core.LoDTensor()
        rank_table_tensor.set_recursive_sequence_lengths([[1, 2, 3]])
        rank_table_tensor.set(
            np.random.random(size=(6, 1)).astype('float32'), cpu)

        exe = Executor(cpu)
        outs = exe.run(
            feed={
                'x': x_tensor,
                'rank_table_tensor': rank_table_tensor
            },
            fetch_list=[self.mem1, self.mem2, self.mem3, self.x_grad],
            return_numpy=False)
        np.testing.assert_allclose(tensor_np[0:3], outs[0], rtol=1e-05)
        np.testing.assert_allclose(tensor_np[0:2], outs[1], rtol=1e-05)
        np.testing.assert_allclose(tensor_np[0:1], outs[2], rtol=1e-05)
        self.assertAlmostEqual(1.0, self.sum_lodtensor(outs[3]), delta=0.01)


class TestShrinkRNNMemoryOpError(unittest.TestCase):

    def test_erroes(self):
        with program_guard(Program(), Program()):
            x = layers.zeros(dtype='int64', shape=[3, 100])
            i = layers.zeros(dtype='int64', shape=[1])
            rank_table_tensor = core.LoDTensor()
            rank_table_tensor.set_recursive_sequence_lengths([[1, 2, 3]])
            rank_table_tensor.set(
                np.random.random(size=(6, 1)).astype('float32'),
                core.CPUPlace())
            rank_table = np.random.random(size=(6, 1)).astype('float32')

            # The type of x in shrink_rnn_memory must be Variable.
            def test_x_type():
                out = shrink_memory(x=1, i=i, table=rank_table_tensor)

            self.assertRaises(TypeError, test_x_type)

            # The type of i in shrink_rnn_memory must be Variable.
            def test_i_type():
                out = shrink_memory(x=x, i=0, table=rank_table_tensor)

            self.assertRaises(TypeError, test_i_type)

            # The type of table in shrink_rnn_memory must be Variable.
            def test_table_type():
                out = shrink_memory(x=x, i=i, table=rank_table)

            self.assertRaises(TypeError, test_table_type)


if __name__ == '__main__':
    unittest.main()
