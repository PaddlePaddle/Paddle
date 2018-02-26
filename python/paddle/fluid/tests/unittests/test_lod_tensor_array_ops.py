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
import paddle.fluid.core as core
import numpy
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward


class TestCPULoDTensorArrayOps(unittest.TestCase):
    def place(self):
        return core.CPUPlace()

    def test_lod_tensor_to_array_level_0(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(10).reshape(10, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 3, 9, 10]])
        expect = map(lambda x: numpy.array(x).astype('int32'),
                     [[3, 0, 9], [4, 1], [5, 2], [6], [7], [8]])
        self.main(
            tensor=tensor,
            expect_array=expect,
            expect_lod=[] * 6,
            expect_max_len=6)

    def test_lod_tensor_to_array_level_0_empty_seq(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(10).reshape(10, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 3, 9, 9, 10]])
        expect = map(lambda x: numpy.array(x).astype('int32'),
                     [[3, 0, 9], [4, 1], [5, 2], [6], [7], [8]])
        self.main(
            tensor=tensor,
            expect_array=expect,
            expect_lod=[] * 6,
            expect_max_len=6)

    def test_lod_tensor_to_array_level_1(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(20).reshape(20, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 2, 5], [0, 3, 9, 11, 17, 20]])

        expect = [
            numpy.array(
                [9, 10, 0, 1, 2], dtype='int32'), numpy.array(
                    [11, 12, 13, 14, 15, 16, 3, 4, 5, 6, 7, 8], dtype='int32'),
            numpy.array(
                [17, 18, 19], dtype='int32')
        ]

        lod = [[[0, 2, 5]], [[0, 6, 12]], [[0, 3]]]
        self.main(
            tensor=tensor,
            expect_array=expect,
            expect_lod=lod,
            expect_max_len=3)

    def test_lod_tensor_to_array_level_1_empty_seq(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(31).reshape(31, 1).astype('int32'), self.place())

        tensor.set_lod([[0, 3, 5, 9, 11],
                        [0, 3, 7, 11, 11, 12, 17, 19, 21, 23, 30, 31]])

        expect = [
            numpy.array(
                item, dtype='int32')
            for item in [[
                12, 13, 14, 15, 16, 0, 1, 2, 23, 24, 25, 26, 27, 28, 29
            ], [17, 18, 3, 4, 5, 6, 11, 30], [19, 20, 7, 8, 9, 10], [21, 22]]
        ]

        lod = [[[0, 5, 8, 8, 15]], [[0, 2, 6, 7, 8]], [[0, 2, 6]], [[0, 2]]]
        self.main(
            tensor=tensor,
            expect_array=expect,
            expect_lod=lod,
            expect_max_len=4)

    def test_lod_tensor_to_array_level_2(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(50).reshape(50, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 2, 5, 6], [0, 2, 5, 6, 10, 12, 13],
                        [0, 3, 7, 11, 17, 21, 22, 23, 27, 31, 39, 45, 46, 50]])

        expect = [
            numpy.array(
                item, dtype='int32')
            for item in [[21, 0, 1, 2, 3, 4, 5, 6, 46, 47, 48, 49], range(
                22, 39) + range(7, 21), range(39, 46)]
        ]
        lod = [[[0, 1, 3, 4], [0, 1, 4, 8, 12]],
               [[0, 4, 7], [0, 1, 5, 9, 17, 21, 27, 31]], [[0, 2], [0, 6, 7]]]
        self.main(
            tensor=tensor,
            expect_array=expect,
            expect_lod=lod,
            expect_max_len=3)

    def test_lod_tensor_to_array_level_2_skip_level(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(50).reshape(50, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 2, 5, 6], [0, 2, 5, 6, 10, 12, 13],
                        [0, 3, 7, 11, 17, 21, 22, 23, 27, 31, 39, 45, 46, 50]])
        self.main(
            tensor=tensor,
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
            table = layers.lod_rank_table(x, level=level)
            max_len = layers.max_sequence_len(table)
            max_len.persistable = True
            array = layers.lod_tensor_to_array(x, table)
            array.persistable = True

            result = layers.array_to_lod_tensor(array, table)
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
            numpy.array(scope.find_var(max_len.name).get_tensor())[0],
            expect_max_len)

    def check_array_same(self, array, expect_tensor, expect_lod):
        self.assertEqual(len(expect_tensor), len(array))
        for i, exp in enumerate(zip(expect_tensor, expect_lod)):
            exp_tensor, exp_lod = exp
            exp_tensor = numpy.expand_dims(exp_tensor, axis=1)
            self.assertTrue(numpy.allclose(exp_tensor, numpy.array(array[i])))
            self.assertEqual(exp_lod, array[i].lod())

    def check_tensor_same(self, actual, expect):
        self.assertTrue(
            numpy.allclose(numpy.array(actual), numpy.array(expect)))
        self.assertEqual(actual.lod(), expect.lod())


class TestCPULoDTensorArrayOpGrad(unittest.TestCase):
    def test_grad(self):
        place = core.CPUPlace()
        program = Program()

        with program_guard(program):
            x = layers.data(
                name='x', shape=[1], dtype='float32', stop_gradient=False)
            table = layers.lod_rank_table(x, level=0)
            array = layers.lod_tensor_to_array(x, table)
            result = layers.array_to_lod_tensor(array, table)

            mean = layers.mean(result)

            append_backward(mean)

        tensor = core.LoDTensor()
        tensor.set(numpy.arange(10).reshape(10, 1).astype('float32'), place)
        tensor.set_lod([[0, 3, 9, 10]])

        g_vars = program.global_block().var(x.name + "@GRAD")

        exe = Executor(place)
        g_out = [
            numpy.array(item).sum()
            for item in exe.run(program,
                                feed={'x': tensor},
                                fetch_list=[g_vars],
                                return_numpy=False)
        ]
        g_out_sum = numpy.array(g_out).sum()

        self.assertAlmostEqual(1.0, g_out_sum, delta=0.1)


if __name__ == '__main__':
    unittest.main()
