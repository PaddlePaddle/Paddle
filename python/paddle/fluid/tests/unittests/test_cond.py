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
            return 1, 2
        else:
            return 3, 4
        """

        def true_func():
            return layers.fill_constant(
                shape=[1, 2], dtype='int32', value=1), layers.fill_constant(
                    shape=[2, 3], dtype='float32', value=2)

        def false_func():
            return layers.fill_constant(
                shape=[3, 4], dtype='int32', value=3), layers.fill_constant(
                    shape=[4, 5], dtype='float32', value=4)

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
            np.allclose(np.asarray(ret[1]), np.full((2, 3), 2, np.float32)))

    def test_pass_and_modify_var(self):
        """
        pseudocode:
        for i in range(5):
            if i % 2 == 0:
                return 1, 2
        else:
            return 3, 4
        """

        def true_func(a):
            a = a + 2
            return a

        def false_func(a):
            a = a - 1
            return a

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            pred = fluid.data(name="condition", shape=[1], dtype='bool')
            a = layers.fill_constant(shape=[3, 2, 1], dtype='int', value=1)
            out = layers.cond(pred, true_func, false_func)


if __name__ == '__main__':
    unittest.main()
