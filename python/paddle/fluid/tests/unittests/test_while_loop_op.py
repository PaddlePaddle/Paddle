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
from paddle.fluid.backward import append_backward


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

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
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
            mem = layers.data(
                name='mem',
                shape=[10],
                dtype='float32',
                append_batch_size=False)
            one = layers.fill_constant(shape=[10], dtype='float32', value=1)
            out = layers.while_loop(cond, body, [i, mem])

            data = np.random.rand(10).astype('float32')
            data_one = np.ones(10).astype('float32')

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
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
            init = layers.data(
                name='init',
                shape=[3, 3],
                dtype='float32',
                append_batch_size=False)
            sums = layers.data(
                name='sums',
                shape=[3, 3],
                dtype='float32',
                append_batch_size=False)
            loop_len1 = layers.fill_constant(shape=[1], dtype='int64', value=2)
            loop_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
            ones = layers.fill_constant(shape=[3, 3], dtype='float32', value=1)

            out = layers.while_loop(external_cond, external_body,
                                    [i, j, init, sums])

            data = np.random.rand(3, 3).astype('float32')
            data_sums = np.zeros([3, 3]).astype('float32')

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program,
                      feed={'init': data,
                            'sums': data_sums},
                      fetch_list=out)
        for i in range(3):
            data = np.add(data, 1)
            data_sums = np.add(data, data_sums)
        for j in range(2):
            data_sums = np.add(data, data_sums)
        self.assertTrue(np.allclose(np.asarray(res[3]), data_sums))


class TestApiWhileLoop_Backward(unittest.TestCase):
    def test_while_loop_backward(self):
        def cond(i, x):
            return layers.less_than(i, eleven)

        def body(i, x):
            x = layers.elementwise_mul(x=i, y=i)
            i = layers.increment(i)
            return [i, x]

        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            i = layers.data(
                name='i', shape=[1], dtype='float32', append_batch_size=False)
            i.stop_gradient = False
            eleven = layers.fill_constant(shape=[1], dtype='float32', value=11)
            one = layers.fill_constant(shape=[1], dtype='float32', value=1)
            x = layers.data(
                name='x', shape=[1], dtype='float32', append_batch_size=False)
            x.stop_gradient = False

            out = layers.while_loop(cond, body, [i, x])
            mean = layers.mean(out[1])
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        feed_i = np.ones(1).astype('float32')
        feed_x = np.ones(1).astype('float32')
        data = np.asarray([100]).astype('float32')
        i_grad = np.asarray([110]).astype('float32')

        res = exe.run(main_program,
                      feed={'i': feed_i,
                            'x': feed_x},
                      fetch_list=[mean.name, i.grad_name])
        self.assertTrue(np.allclose(np.asarray(res[0]), data))
        self.assertTrue(np.allclose(np.asarray(res[1]), i_grad))


class TestApiWhileLoop_NestedWithBackward(unittest.TestCase):
    def test_nested_net_with_backward(self):
        def external_cond(i, x, y):
            return layers.less_than(i, ten)

        def external_body(i, x, y):
            def internal_cond(i, x, y):
                return layers.less_than(i, five)

            def internal_body(i, x, y):
                x = layers.elementwise_add(x=i, y=i)
                i = layers.increment(i)
                return [i, x, y]

            temp = layers.while_loop(internal_cond, internal_body, [i, x, y])
            y = layers.elementwise_add(x=temp[1], y=i)
            i = layers.increment(i)
            return [i, x, y]

        main_program = Program()
        startup_program = Program()

        with fluid.program_guard(main_program, startup_program):
            i = layers.data(
                name='i', shape=[1], dtype='float32', append_batch_size=False)
            i.stop_gradient = False
            ten = layers.fill_constant(shape=[1], dtype='float32', value=10)
            five = layers.fill_constant(shape=[1], dtype='float32', value=5)
            x = layers.data(
                name='x', shape=[1], dtype='float32', append_batch_size=False)
            x.stop_gradient = False
            y = layers.data(
                name='y', shape=[1], dtype='float32', append_batch_size=False)
            y.stop_gradient = False
            out = layers.while_loop(external_cond, external_body, [i, x, y])

            mean = layers.mean(out[2])
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        data = np.asarray([17]).astype('float32')
        feed_x = np.zeros(1).astype('float32')
        feed_i = np.ones(1).astype('float32')
        feed_y = np.zeros(1).astype('float32')
        i_grad = np.asarray(13).astype('int32')

        res = exe.run(main_program,
                      feed={'i': feed_i,
                            'x': feed_x,
                            'y': feed_y},
                      fetch_list=[mean.name, i.grad_name])

        self.assertTrue(np.allclose(np.asarray(res[0]), data))
        self.assertTrue(np.allclose(np.asarray(res[1]), i_grad))


class TestApiWhileLoopWithSwitchCase(unittest.TestCase):
    def test_with_switch_case(self):
        def cond(i):
            return layers.less_than(i, ten)

        def body(i):
            def fn_add_three():
                data_add_three = layers.elementwise_add(x=i, y=three)
                return data_add_three

            def fn_square():
                data_mul_data = layers.elementwise_mul(x=i, y=i)
                return data_mul_data

            def fn_add_one():
                data_add_one = layers.elementwise_add(x=i, y=one)
                return data_add_one

            return layers.switch_case(
                branch_index=i,
                branch_fns={2: fn_add_three,
                            5: fn_square},
                default=fn_add_one)

        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            i = layers.fill_constant(shape=[1], dtype='int64', value=1)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            three = layers.fill_constant(shape=[1], dtype='int64', value=3)
            one = layers.fill_constant(shape=[1], dtype='int64', value=1)
            out = layers.while_loop(cond, body, [i])

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, fetch_list=out)

        data = np.asarray([25]).astype('int64')
        self.assertTrue(np.allclose(np.asarray(res[0]), data))


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
