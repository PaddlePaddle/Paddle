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
import numpy as np
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward

from paddle.fluid.layers.control_flow import lod_rank_table
from paddle.fluid.layers.control_flow import max_sequence_len
from paddle.fluid.layers.control_flow import lod_tensor_to_array
from paddle.fluid.layers.control_flow import array_to_lod_tensor


class TestCPULoDTensorArrayOps(unittest.TestCase):

    def place(self):
        return core.CPUPlace()

    def test_lod_tensor_to_array_level_0(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(10).reshape(10, 1).astype('int32'), self.place())
        tensor.set_recursive_sequence_lengths([[3, 6, 1]])
        expect = [
            np.array(x).astype('int32')
            for x in [[3, 0, 9], [4, 1], [5, 2], [6], [7], [8]]
        ]
        self.main(tensor=tensor,
                  expect_array=expect,
                  expect_lod=[] * 6,
                  expect_max_len=6)

    def test_lod_tensor_to_array_level_0_empty_seq(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(10).reshape(10, 1).astype('int32'), self.place())
        tensor.set_recursive_sequence_lengths([[3, 6, 0, 1]])
        expect = [
            np.array(x).astype('int32')
            for x in [[3, 0, 9], [4, 1], [5, 2], [6], [7], [8]]
        ]
        self.main(tensor=tensor,
                  expect_array=expect,
                  expect_lod=[] * 6,
                  expect_max_len=6)

    def test_lod_tensor_to_array_level_1(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(20).reshape(20, 1).astype('int32'), self.place())
        tensor.set_recursive_sequence_lengths([[2, 3], [3, 6, 2, 6, 3]])

        expect = [
            np.array([9, 10, 0, 1, 2], dtype='int32'),
            np.array([11, 12, 13, 14, 15, 16, 3, 4, 5, 6, 7, 8], dtype='int32'),
            np.array([17, 18, 19], dtype='int32')
        ]

        lod = [[[2, 3]], [[6, 6]], [[3]]]
        self.main(tensor=tensor,
                  expect_array=expect,
                  expect_lod=lod,
                  expect_max_len=3)

    def test_lod_tensor_to_array_level_1_empty_seq(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(31).reshape(31, 1).astype('int32'), self.place())

        tensor.set_recursive_sequence_lengths(
            [[3, 2, 4, 2], [3, 4, 4, 0, 1, 5, 2, 2, 2, 7, 1]])

        expect = [
            np.array(item, dtype='int32') for item in [[
                12, 13, 14, 15, 16, 0, 1, 2, 23, 24, 25, 26, 27, 28, 29
            ], [17, 18, 3, 4, 5, 6, 11, 30], [19, 20, 7, 8, 9, 10], [21, 22]]
        ]

        lod = [[[5, 3, 0, 7]], [[2, 4, 1, 1]], [[2, 4]], [[2]]]
        self.main(tensor=tensor,
                  expect_array=expect,
                  expect_lod=lod,
                  expect_max_len=4)

    def test_lod_tensor_to_array_level_2(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(50).reshape(50, 1).astype('int32'), self.place())
        tensor.set_recursive_sequence_lengths(
            [[2, 3, 1], [2, 3, 1, 4, 2, 1],
             [3, 4, 4, 6, 4, 1, 1, 4, 4, 8, 6, 1, 4]])

        expect = [
            np.array(item, dtype='int32')
            for item in [[21, 0, 1, 2, 3, 4, 5, 6, 46, 47, 48, 49],
                         list(range(22, 39)) + list(range(7, 21)),
                         list(range(39, 46))]
        ]
        lod = [[[1, 2, 1], [1, 3, 4, 4]], [[4, 3], [1, 4, 4, 8, 4, 6, 4]],
               [[2], [6, 1]]]
        self.main(tensor=tensor,
                  expect_array=expect,
                  expect_lod=lod,
                  expect_max_len=3)

    def test_lod_tensor_to_array_level_2_skip_level(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(50).reshape(50, 1).astype('int32'), self.place())
        tensor.set_recursive_sequence_lengths(
            [[2, 3, 1], [2, 3, 1, 4, 2, 1],
             [3, 4, 4, 6, 4, 1, 1, 4, 4, 8, 6, 1, 4]])
        self.main(tensor=tensor,
                  expect_array=None,
                  expect_lod=None,
                  expect_max_len=4,
                  level=1)

    def main(self, tensor, expect_array, expect_lod, expect_max_len, level=0):
        place = self.place()
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[10])
            x.persistable = True
            table = lod_rank_table(x, level=level)
            max_len = max_sequence_len(table)
            max_len.persistable = True
            array = lod_tensor_to_array(x, table)
            array.persistable = True

            result = array_to_lod_tensor(array, table)
            result.persistable = True
        exe = Executor(place)
        scope = core.Scope()
        exe.run(program, feed={'x': tensor}, scope=scope)
        var = scope.find_var(array.name)
        array = var.get_lod_tensor_array()
        if expect_array is not None and expect_lod is not None:
            self.check_array_same(array, expect_array, expect_lod)
        self.check_tensor_same(scope.find_var(result.name).get_tensor(), tensor)

        self.assertEqual(
            np.array(scope.find_var(max_len.name).get_tensor())[0],
            expect_max_len)

    def check_array_same(self, array, expect_tensor, expect_lod):
        self.assertEqual(len(expect_tensor), len(array))
        for i, exp in enumerate(zip(expect_tensor, expect_lod)):
            exp_tensor, exp_lod = exp
            exp_tensor = np.expand_dims(exp_tensor, axis=1)
            np.testing.assert_allclose(exp_tensor,
                                       np.array(array[i]),
                                       rtol=1e-05)
            self.assertEqual(exp_lod, array[i].recursive_sequence_lengths())

    def check_tensor_same(self, actual, expect):
        np.testing.assert_allclose(np.array(actual),
                                   np.array(expect),
                                   rtol=1e-05)
        self.assertEqual(actual.recursive_sequence_lengths(),
                         expect.recursive_sequence_lengths())


class TestCPULoDTensorArrayOpGrad(unittest.TestCase):

    def test_grad(self):
        place = core.CPUPlace()
        program = Program()

        with program_guard(program):
            x = layers.data(name='x',
                            shape=[1],
                            dtype='float32',
                            stop_gradient=False)
            table = lod_rank_table(x, level=0)
            array = lod_tensor_to_array(x, table)
            result = array_to_lod_tensor(array, table)

            mean = paddle.mean(result)

            append_backward(mean)

        tensor = core.LoDTensor()
        tensor.set(np.arange(10).reshape(10, 1).astype('float32'), place)
        tensor.set_recursive_sequence_lengths([[3, 6, 1]])

        g_vars = program.global_block().var(x.name + "@GRAD")

        exe = Executor(place)
        g_out = [
            np.array(item).sum() for item in exe.run(program,
                                                     feed={'x': tensor},
                                                     fetch_list=[g_vars],
                                                     return_numpy=False)
        ]
        g_out_sum = np.array(g_out).sum()

        self.assertAlmostEqual(1.0, g_out_sum, delta=0.1)


class TestLoDTensorArrayError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            x = np.random.random((10)).astype("float32")
            x2 = layers.data(name='x', shape=[10])
            table = lod_rank_table(x2, level=0)

            def test_x_Variable():
                rank_table = lod_tensor_to_array(x=x, table=table)

            self.assertRaises(TypeError, test_x_Variable)

            table2 = np.random.random((2)).astype("int64")

            def test_table_Variable():
                rank_table = lod_tensor_to_array(x=x2, table=table2)

            self.assertRaises(TypeError, test_table_Variable)

            def test_x_list_Variable():
                rank_table = lod_tensor_to_array(x=[x], table=table)

            self.assertRaises(TypeError, test_x_list_Variable)

            def test_table_list_Variable():
                rank_table = lod_tensor_to_array(x=x2, table=[table2])

            self.assertRaises(TypeError, test_table_list_Variable)

            array = lod_tensor_to_array(x2, table)


class TestArrayLoDTensorError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            x = np.random.random((10)).astype("float32")
            x2 = layers.data(name='x', shape=[10])
            table = lod_rank_table(x2, level=0)
            array = lod_tensor_to_array(x2, table)

            def test_x_Variable():
                rank_table = array_to_lod_tensor(x=x, table=table)

            self.assertRaises(TypeError, test_x_Variable)

            table2 = np.random.random((2)).astype("int64")

            def test_table_Variable():
                rank_table = array_to_lod_tensor(x=array, table=table2)

            self.assertRaises(TypeError, test_table_Variable)

            def test_x_list_Variable():
                rank_table = array_to_lod_tensor(x=[x], table=table)

            self.assertRaises(TypeError, test_x_list_Variable)

            def test_table_list_Variable():
                rank_table = array_to_lod_tensor(x=x2, table=[table2])

            self.assertRaises(TypeError, test_table_list_Variable)

            array = array_to_lod_tensor(x2, table)


if __name__ == '__main__':
    unittest.main()
