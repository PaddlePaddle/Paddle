#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict


class TestAPICase(unittest.TestCase):
    def test_return_single_var(self):
        def fn_1():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=1)

        def fn_2():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=2)

        def fn_3():
            return layers.fill_constant(shape=[4, 3], dtype='int32', value=3)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)
            pred_2 = layers.less_than(x, y)
            pred_1 = layers.less_than(z, x)  # true

            # call fn_1
            out_0 = layers.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_1, fn_2)], default=fn_3)

            # call fn_2
            out_1 = layers.case(
                pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)], default=fn_3)

            # call default fn_3
            out_2 = layers.case(
                pred_fn_pairs=((pred_2, fn_1), (pred_2, fn_2)), default=fn_3)

            # no default, call fn_2
            out_3 = layers.case(pred_fn_pairs=[(pred_1, fn_2)])

            # no default, call fn_2. but pred_2 is false
            out_4 = layers.case(pred_fn_pairs=[(pred_2, fn_2)])

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)

            res = exe.run(main_program,
                          fetch_list=[out_0, out_1, out_2, out_3, out_4])

            self.assertTrue(np.allclose(res[0], 1))
            self.assertTrue(np.allclose(res[1], 2))
            self.assertTrue(np.allclose(res[2], 3))
            self.assertTrue(np.allclose(res[3], 2))
            self.assertTrue(np.allclose(res[4], 2))

    def test_return_var_tuple(self):
        def fn_1():
            return layers.fill_constant(
                shape=[1, 2], dtype='int32', value=1), layers.fill_constant(
                    shape=[2, 3], dtype='float32', value=2)

        def fn_2():
            return layers.fill_constant(
                shape=[3, 4], dtype='int32', value=3), layers.fill_constant(
                    shape=[4, 5], dtype='float32', value=4)

        def fn_3():
            return layers.fill_constant(
                shape=[5], dtype='int32', value=5), layers.fill_constant(
                    shape=[5, 6], dtype='float32', value=6)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=1)
            y = layers.fill_constant(shape=[1], dtype='float32', value=1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=3)

            pred_1 = layers.equal(x, y)  # true
            pred_2 = layers.equal(x, z)  # false

            out = layers.case(((pred_1, fn_1), (pred_2, fn_2)), fn_3)

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            ret = exe.run(main_program, fetch_list=out)

            self.assertTrue(
                np.allclose(np.asarray(ret[0]), np.full((1, 2), 1, np.int32)))
            self.assertTrue(
                np.allclose(
                    np.asarray(ret[1]), np.full((2, 3), 2, np.float32)))

    def test_dic(self):
        def fn_1():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=1)

        def fn_2():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=2)

        def fn_3():
            return layers.fill_constant(shape=[4, 3], dtype='int32', value=3)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)
            pred_2 = layers.less_than(y, x)  # true
            pred_1 = layers.less_than(z, x)  # true

            pred_fn_pairs = OrderedDict({pred_1: fn_1, pred_2: fn_2})
            out_0 = layers.case(pred_fn_pairs, default=fn_3)

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)

            res = exe.run(main_program, fetch_list=[out_0])
            #print(res)
            self.assertTrue(np.allclose(res[0], 1))


class TestAPICase_Error(unittest.TestCase):
    def test_error(self):
        def fn_1():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=1)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)
            pred_1 = layers.less_than(z, x)  # true

            # The type of 'pred_fn_pairs' in case must be list, tuple or dict
            def type_error_pred_fn_pairs():
                layers.case(pred_fn_pairs=1, default=fn_1)

            self.assertRaises(TypeError, type_error_pred_fn_pairs)

            # The elements' type of 'pred_fn_pairs' in Op(case) must be tuple
            def type_error_pred_fn_1():
                layers.case(pred_fn_pairs=[1], default=fn_1)

            self.assertRaises(TypeError, type_error_pred_fn_1)

            # The tuple's size of 'pred_fn_pairs' in Op(case) must be 2
            def type_error_pred_fn_2():
                layers.case(pred_fn_pairs=[(1, 2, 3)], default=fn_1)

            self.assertRaises(TypeError, type_error_pred_fn_2)

            # The pred's type of 'pred_fn_pairs' in Op(case) must be bool Variable
            def type_error_pred():
                layers.case(pred_fn_pairs=[(1, 2)], default=fn_1)

            self.assertRaises(TypeError, type_error_pred)

            # The pred's data type of 'pred_fn_pairs' in Op(case) must be bool
            def dtype_error_pred():
                layers.case(pred_fn_pairs=[(x, 2)], default=fn_1)

            self.assertRaises(TypeError, dtype_error_pred)

            # The function of pred_fn_pairs in case must be callable
            def type_error_fn():
                layers.case(pred_fn_pairs=[(pred_1, 2)], default=fn_1)

            self.assertRaises(TypeError, type_error_fn)

            # The default in Op(case) must be callable
            def type_error_default():
                layers.case(pred_fn_pairs=[(pred_1, fn_1)], default=fn_1())

            self.assertRaises(TypeError, type_error_default)


if __name__ == '__main__':
    unittest.main()
