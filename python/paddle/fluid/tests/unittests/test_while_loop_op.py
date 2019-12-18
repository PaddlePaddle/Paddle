# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class TestApiWhileLoop(unittest.TestCase):
    def test_var_tuple(self):
        def cond(i):
            return layers.less_than(i, ten)

        def body(i):
            return layers.elementwise_add(x=i, y=one)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.fill_constant(shape=[1], dtype='int64', value=0)
            one = layers.fill_constant(shape=[1], dtype='int64', value=1)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            out = layers.while_loop(cond, body, (i, ))

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, fetch_list=out)
        self.assertTrue(
            np.allclose(np.asarray(res[0]), np.full((1), 10, np.int64)))

    def test_var_list(self):
        def cond(i, mem):
            return layers.less_than(i, ten)

        def body(i, mem):
            mem = layers.elementwise_add(x=mem, y=one)
            i = layers.increment(i)
            return [i, mem]

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.zeros(shape=[1], dtype='int64')
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            mem = layers.data(name="mem", shape=[10], dtype='float32')
            one = layers.fill_constant(shape=[10], dtype='float32', value=1)
            out = layers.while_loop(cond, body, [i, mem])

            data = np.random.rand(10).astype('float32')
            data_one = np.ones(10).astype('float32')

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, feed={'mem': data}, fetch_list=out)
        for i in range(10):
            data = np.add(data, data_one)
        self.assertTrue(np.allclose(np.asarray(res[1]), data))


class TestApiWhileLoop_Nested(unittest.TestCase):
    def test_nested_net(self):
        def external_cond(i, j, init, sums):
            return layers.less_than(i, loop_len1)

        def external_body(i, j, init, sums):
            def internal_cond(j, init, sums):
                return layers.less_than(j, loop_len2)

            def internal_body(j, init, sums):
                init = layers.elementwise_add(x=init, y=ones)
                sums = layers.elementwise_add(x=init, y=sums)
                j = layers.increment(j)
                return [j, init, sums]

            result = layers.while_loop(internal_cond, internal_body,
                                       [j, init, sums])
            j = result[0]
            init = result[1]
            sums = result[2]
            sums = layers.elementwise_add(x=init, y=sums)
            i = layers.increment(i)
            return [i, j, init, sums]

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.zeros(shape=[1], dtype='int64')
            j = layers.zeros(shape=[1], dtype='int64')
            init = layers.data(name="init", shape=[3, 3], dtype='float32')
            sums = layers.data(name="sums", shape=[3, 3], dtype='float32')
            loop_len1 = layers.fill_constant(shape=[1], dtype='int64', value=2)
            loop_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
            ones = layers.fill_constant(shape=[3, 3], dtype='float32', value=1)

            res = layers.while_loop(external_cond, external_body,
                                    [i, j, init, sums])

            data = np.random.rand(3, 3).astype('float32')
            data_sums = np.zeros([3, 3]).astype('float32')

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program,
                      feed={'init': data,
                            'sums': data_sums},
                      fetch_list=res)
        for i in range(3):
            data = np.add(data, 1)
            data_sums = np.add(data, data_sums)
        for j in range(2):
            data_sums = np.add(data, data_sums)
        self.assertTrue(np.allclose(np.asarray(ret[3]), data_sums))


class TestApiWhileLoop_Error(unittest.TestCase):
    def test_error(self):
        def cond_returns_constant(i):
            return 1

        def cond_returns_not_bool_tensor(i):
            return layers.increment(i)

        def cond_returns_bool_tensor(i):
            return layers.less_than(i, ten)

        def cond_returns_2d_tensor(i):
            return layers.less_than(i, ten_2d)

        def body(i):
            return layers.increment(i)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            data = layers.fill_constant(shape=[1], dtype='int64', value=1)
            data_1d = layers.fill_constant(shape=[1], dtype='int64', value=1)
            data_2d = layers.fill_constant(shape=[2, 2], dtype='int64', value=1)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            ten_2d = layers.fill_constant(shape=[2, 2], dtype='int64', value=10)

            # The type of `cond` in Op(while_loop) must be callable 
            def type_error_cond():
                out = layers.while_loop(data, body, [data_1d])

            self.assertRaises(TypeError, type_error_cond)

            # The type of `body` in Op(while_loop) must be callable
            def type_error_body():
                out = layers.while_loop(cond_returns_bool_tensor, data,
                                        [data_1d])

            self.assertRaises(TypeError, type_error_body)

            # The type of `loop_vars` in Op(while_loop) must be list or tuple
            def type_error_loop_vars():
                out = layers.while_loop(cond_returns_bool_tensor, body, data_1d)

            self.assertRaises(TypeError, type_error_loop_vars)

            # The value of `loop_vars` is empty
            def value_error_loop_vars():
                out = layers.while_loop(cond_returns_bool_tensor, body, [])

            self.assertRaises(ValueError, value_error_loop_vars)

            # The type of `cond` returns in Op(while_loop) must be Variable
            def type_error_cond_returns_not_variable():
                out = layers.while_loop(cond_returns_constant, body, [data_1d])

            self.assertRaises(TypeError, type_error_cond_returns_not_variable)

            # The type of `cond` returns in Op(while_loop) must be a bollean variable
            def type_error_cond_returns_not_boolean():
                out = layers.while_loop(cond_returns_not_bool_tensor, body,
                                        [data_1d])

            self.assertRaises(TypeError, type_error_cond_returns_not_boolean)

            # The shape of `cond` returns in Op(while_loop) must be 1
            def type_error_shape_cond_returns_2d():
                out = layers.while_loop(cond_returns_2d_tensor, body, [data_2d])

            self.assertRaises(TypeError, type_error_shape_cond_returns_2d)


if __name__ == '__main__':
    unittest.main()
