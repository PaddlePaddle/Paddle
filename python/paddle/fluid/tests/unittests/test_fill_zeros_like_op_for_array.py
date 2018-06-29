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

import paddle.fluid as fluid
import paddle.fluid.core as core


class TestFillZerosLikeOpForTensorArray(unittest.TestCase):
    def place(self):
        return core.CPUPlace()

    def test_zero_filling_lod_tensor_array(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(20).reshape(20, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 2, 5], [0, 3, 9, 11, 17, 20]])

        expect = [
            numpy.array(
                [0, 0, 0, 0, 0], dtype='int32'), numpy.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int32'),
            numpy.array(
                [0, 0, 0], dtype='int32')
        ]

        lod = [[[0, 2, 5]], [[0, 6, 12]], [[0, 3]]]
        self.main(
            tensor=tensor,
            expect_array=expect,
            expect_lod=lod,
            expect_max_len=3)

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
            array = layers.fill_zeros_like(array)
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


if __name__ == '__main__':
    unittest.main()
