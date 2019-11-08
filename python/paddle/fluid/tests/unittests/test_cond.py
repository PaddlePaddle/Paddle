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

from __future__ import print_function

import numpy as np
import unittest

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, program_guard


class TestCond(unittest.TestCase):
    def test_return_single_var(self):
        """
        pseudocode:

        if 0.23 < 0.1:
            return 2
        else:
            return -1
        """

        def true_func():
            return layers.fill_constant(shape=[2, 3], dtype='int32', value=2)

        def false_func():
            return layers.fill_constant(shape=[3, 2], dtype='int32', value=-1)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
            pred = layers.less_than(y, x)
            out = layers.cond(pred, true_func, false_func)
            # out is one tensor

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program, fetch_list=[out.name])
        self.assertTrue(
            np.allclose(np.asarray(ret), np.full((3, 2), -1, np.int32)))

    def test_return_var_tuple(self):
        """
        pseudocode:

        if True:
            return 1, True
        else:
            return 3, 2
        """

        def true_func():
            return layers.fill_constant(
                shape=[1, 2], dtype='int32', value=1), layers.fill_constant(
                    shape=[2, 3], dtype='bool', value=True)

        def false_func():
            return layers.fill_constant(
                shape=[3, 4], dtype='float32', value=3), layers.fill_constant(
                    shape=[4, 5], dtype='int64', value=2)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            pred = layers.fill_constant(shape=[1], dtype='bool', value=True)
            out = layers.cond(pred, true_func, false_func)
            # out is a tuple containing 2 tensors

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program, fetch_list=out)
        self.assertTrue(
            np.allclose(np.asarray(ret[0]), np.full((1, 2), 1, np.int32)))
        self.assertTrue(
            np.allclose(np.asarray(ret[1]), np.full((2, 3), True, np.bool)))

    def test_pass_and_modify_var(self):
        """
        pseudocode:
        for i in range(5):
            a = 7
            if i % 2 == 0:
                a = a * (i + 1)
            else:
                a = a - (i - 1)
        """

        def true_func(a, i):
            a = a * (i + 1)
            return a

        def false_func(a, i):
            a = a - (i - 1)
            return a

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            a = layers.fill_constant(shape=[3, 2, 1], dtype='int32', value=7)
            i = fluid.data(name="i", shape=[1], dtype='int32')
            pred = ((i % 2) == 0)
            a = layers.cond(pred, lambda: true_func(a, i),
                            lambda: false_func(a, i))
        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        for feed_i in range(5):
            expected_a = 7 * (feed_i + 1) if feed_i % 2 == 0 else 8 - feed_i
            ret = exe.run(main_program,
                          feed={'i': np.full((1), feed_i, np.int32)},
                          fetch_list=[a])
            self.assertTrue(
                np.allclose(
                    np.asarray(ret), np.full((3, 2, 1), expected_a, np.int32)))

    def test_return_none(self):
        """
        pseudocode: test doing nothing in branches
        for i in range(5):
            if i % 2 == 0:
                pass
            else:
                pass
        """

        def true_func():
            pass

        def false_func():
            return None

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = fluid.data(name="i", shape=[1], dtype='int32')
            pred = ((i % 2) == 0)
            out1 = layers.cond(pred, true_func, false_func)
            out2 = layers.cond(pred, None, false_func)
            out3 = layers.cond(pred, true_func, None)
        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        for feed_i in range(5):
            # Test that output is None is runnable
            exe.run(main_program, feed={'i': np.full((1), feed_i, np.int32)})
            self.assertIsNone(out1)
            self.assertIsNone(out2)
            self.assertIsNone(out3)

    def test_wrong_structure_exception(self):
        """
        test returning different number of tensors cannot merge into output
        """

        def func_return_none():
            return None

        def func_return_one_tensor():
            return layers.fill_constant(shape=[2, 7], dtype='int32', value=3)

        def func_return_two_tensors():
            return layers.fill_constant(
                shape=[3, 1], dtype='int32', value=7), layers.fill_constant(
                    shape=[3, 1], dtype='int32', value=8)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = fluid.data(name="i", shape=[1], dtype='int32')
            pred = ((i % 2) == 0)
            with self.assertRaises(Exception) as e:
                out = layers.cond(pred, i, func_return_one_tensor)
            self.assertEqual("The true_fn in cond must be callable",
                             str(e.exception))

            with self.assertRaises(Exception) as e:
                out = layers.cond(pred, func_return_one_tensor, np.asarray([3]))
            self.assertEqual("The false_fn in cond must be callable",
                             str(e.exception))

            with self.assertRaises(Exception) as e:
                out = layers.cond(pred, func_return_none,
                                  func_return_one_tensor)
            self.assertTrue(
                "Incompatible return values of true_fn and false_fn in cond" in
                str(e.exception))

            with self.assertRaises(Exception) as e:
                out = layers.cond(pred, func_return_two_tensors,
                                  func_return_none)
            self.assertTrue(
                "Incompatible return values of true_fn and false_fn in cond" in
                str(e.exception))

            with self.assertRaises(Exception) as e:
                out = layers.cond(pred, func_return_one_tensor,
                                  func_return_two_tensors)
            self.assertTrue(
                "Incompatible return values of true_fn and false_fn in cond" in
                str(e.exception))


if __name__ == '__main__':
    unittest.main()
