#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from op_test import OpTest

import paddle.fluid.core as core
from paddle import rand
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestRandOpError(unittest.TestCase):
    """
    This class test the input type check.
    """

    def test_errors(self):
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_Variable():
                x1 = fluid.create_lod_tensor(
                    np.zeros((4, 784)), [[1, 1, 1, 1]], fluid.CPUPlace())
                rand(x1)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                dim_1 = fluid.layers.fill_constant([1], "int64", 3)
                dim_2 = fluid.layers.fill_constant([1], "int32", 5)
                rand(shape=[dim_1, dim_2], dtype='int32')

            self.assertRaises(TypeError, test_dtype)

            def test_shape_list():
                rand(shape=[2.])

            self.assertRaises(TypeError, test_shape_list)

            def test_shape_list2():
                rand(shape=[2, 3.])

            self.assertRaises(TypeError, test_shape_list2)

            def test_device():
                rand(shape=[3, 4], device='device')

            self.assertRaises(ValueError, test_device)


class TestRandOp(unittest.TestCase):
    """
    This class test the common usages of randop.

    """

    def test_run(self):
        use_cuda = False
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            result_1 = rand(shape=[3, 4])
            dim_1 = fluid.layers.fill_constant([1], "int64", 3)
            dim_2 = fluid.layers.fill_constant([1], "int32", 5)
            result_2 = rand(shape=[dim_1, dim_2])
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = rand(var_shape)
            var_shape_int32 = fluid.data(
                name='var_shape_int32', shape=[2], dtype="int32")
            result_4 = rand(var_shape_int32)
        exe.run(startup_program)

        x1 = np.array([3, 2]).astype('int64')
        x2 = np.array([4, 3]).astype('int32')
        ret = exe.run(train_program,
                      feed={"var_shape": x1,
                            "var_shape_int32": x2},
                      fetch_list=[result_1, result_2, result_3, result_4])


class TestRandOpForDygraph(unittest.TestCase):
    """
    This class test the common usages of randop.

    """

    def test_run(self):
        use_cuda = False
        with fluid.dygraph.guard():
            rand(shape=[3, 4])
            dim_1 = fluid.layers.fill_constant([1], "int64", 3)
            dim_2 = fluid.layers.fill_constant([1], "int32", 5)
            rand(shape=[dim_1, dim_2])
            var_shape = fluid.dygraph.to_variable(np.array([3, 4]))
            rand(var_shape)


if __name__ == "__main__":
    unittest.main()
