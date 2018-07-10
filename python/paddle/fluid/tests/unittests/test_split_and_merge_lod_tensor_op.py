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
import numpy as np
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward


class TestCPULoDTensorArrayOps(unittest.TestCase):
    def place(self):
        return core.CPUPlace()

    def test_split_and_merge_lod_tensor_no_lod(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(10).reshape(10, 1).astype('int32'), self.place())

        mask_np = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]).astype('bool')
        mask_np = np.expand_dims(mask_np, axis=1)

        mask = core.LoDTensor()
        mask.set(mask_np, self.place())

        expect_true_tensor = np.array([2, 3, 4, 5]).astype('int32')
        expect_true_tensor = np.expand_dims(expect_true_tensor, axis=1)
        expect_true = core.LoDTensor()
        expect_true.set(expect_true_tensor, self.place())

        expect_false_tensor = np.array([0, 1, 6, 7, 8, 9]).astype('int32')
        expect_false_tensor = np.expand_dims(expect_false_tensor, axis=1)

        expect_false = core.LoDTensor()
        expect_false.set(expect_false_tensor, self.place())

        self.main(
            tensor=tensor,
            mask=mask,
            expect_true=expect_true,
            expect_false=expect_false,
            expect_out=tensor)

    def test_split_and_merge_lod_tensor_level_0(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(10).reshape(10, 1).astype('int32'), self.place())
        tensor.set_recursive_sequence_lengths([[3, 6, 1]])

        mask_np = np.array([0, 1, 0]).astype('bool')
        mask_np = np.expand_dims(mask_np, axis=1)

        mask = core.LoDTensor()
        mask.set(mask_np, self.place())

        expect_true_tensor = np.array([3, 4, 5, 6, 7, 8]).astype('int32')
        expect_true_tensor = np.expand_dims(expect_true_tensor, axis=1)
        expect_true = core.LoDTensor()
        expect_true.set(expect_true_tensor, self.place())
        expect_true.set_recursive_sequence_lengths([[6]])

        expect_false_tensor = np.array([0, 1, 2, 9]).astype('int32')
        expect_false_tensor = np.expand_dims(expect_false_tensor, axis=1)
        expect_false_lod = [[3, 1]]

        expect_false = core.LoDTensor()
        expect_false.set(expect_false_tensor, self.place())
        expect_false.set_recursive_sequence_lengths(expect_false_lod)

        self.main(
            tensor=tensor,
            mask=mask,
            expect_true=expect_true,
            expect_false=expect_false,
            expect_out=tensor)

    def main(self, tensor, mask, expect_true, expect_false, expect_out,
             level=0):
        place = self.place()
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[1])
            x.persistable = True

            y = layers.data(name='y', shape=[1])
            y.persistable = True

            out_true, out_false = layers.split_lod_tensor(
                input=x, mask=y, level=level)
            out_true.persistable = True
            out_false.persistable = True

            out = layers.merge_lod_tensor(
                in_true=out_true, in_false=out_false, mask=y, x=x, level=level)

            out.persistable = True

        exe = Executor(place)
        scope = core.Scope()
        exe.run(program,
                feed={'x': tensor,
                      'y': mask},
                scope=scope,
                return_numpy=False)

        var_true = scope.find_var(out_true.name).get_tensor()

        var_false = scope.find_var(out_false.name).get_tensor()

        var_out = scope.find_var(out.name).get_tensor()

        self.check_tensor_same(var_true, expect_true)
        self.check_tensor_same(var_false, expect_false)
        self.check_tensor_same(var_out, expect_out)

    def check_tensor_same(self, actual, expect):
        self.assertTrue(np.allclose(np.array(actual), np.array(expect)))
        self.assertEqual(actual.recursive_sequence_lengths(),
                         expect.recursive_sequence_lengths())


class TestCPUSplitMergeLoDTensorGrad(unittest.TestCase):
    def test_grad(self):
        place = core.CPUPlace()
        program = Program()
        with program_guard(program):
            x = layers.data(
                name='x', shape=[1], dtype='float32', stop_gradient=False)
            y = layers.data(
                name='y', shape=[1], dtype='bool', stop_gradient=False)

            level = 0

            out_true, out_false = layers.split_lod_tensor(
                input=x, mask=y, level=level)
            out = layers.merge_lod_tensor(
                in_true=out_true, in_false=out_false, mask=y, x=x, level=level)
            mean = layers.mean(out)

            append_backward(mean)

        tensor = core.LoDTensor()
        tensor.set(np.arange(10).reshape(10, 1).astype('float32'), place)
        tensor.set_recursive_sequence_lengths([[3, 6, 1]])

        mask_np = np.array([0, 1, 0]).astype('bool')
        mask_np = np.expand_dims(mask_np, axis=1)

        mask = core.LoDTensor()
        mask.set(mask_np, place)

        exe = Executor(place)
        scope = core.Scope()

        g_vars = program.global_block().var(x.name + "@GRAD")
        g_out = [
            item.sum()
            for item in map(np.array,
                            exe.run(program,
                                    feed={'x': tensor,
                                          'y': mask},
                                    fetch_list=[g_vars],
                                    scope=scope,
                                    return_numpy=False))
        ]

        g_out_sum = np.array(g_out).sum()

        self.assertAlmostEqual(1.0, g_out_sum, delta=0.1)


if __name__ == '__main__':
    unittest.main()
