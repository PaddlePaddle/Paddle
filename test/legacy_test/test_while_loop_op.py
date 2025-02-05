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

import unittest

import numpy as np
from utils import compare_legacy_with_pt

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward

paddle.enable_static()


class TestApiWhileLoop(unittest.TestCase):
    @compare_legacy_with_pt
    def test_var_tuple(self):
        def cond(i):
            return paddle.less_than(i, ten)

        def body(i):
            return paddle.add(x=i, y=one)

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
            one = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
            ten = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=10
            )
            out = paddle.static.nn.while_loop(cond, body, (i,))

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        res = exe.run(main_program, fetch_list=out)
        np.testing.assert_allclose(
            np.asarray(res[0]), np.full(1, 10, np.int64), rtol=1e-05
        )

    @compare_legacy_with_pt
    def test_var_list(self):
        def cond(i, mem):
            return paddle.less_than(i, ten)

        def body(i, mem):
            mem = paddle.add(x=mem, y=one)
            i = paddle.increment(i)
            return [i, mem]

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.zeros(shape=[1], dtype='int64')
            ten = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=10
            )
            mem = paddle.static.data(name='mem', shape=[10], dtype='float32')
            one = paddle.tensor.fill_constant(
                shape=[10], dtype='float32', value=1
            )
            out = paddle.static.nn.while_loop(cond, body, [i, mem])

            data = np.random.rand(10).astype('float32')
            data_one = np.ones(10).astype('float32')

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        res = exe.run(main_program, feed={'mem': data}, fetch_list=out)
        for i in range(10):
            data = np.add(data, data_one)
        np.testing.assert_allclose(np.asarray(res[1]), data, rtol=1e-05)

    @compare_legacy_with_pt
    def test_var_dict(self):
        def cond(i, ten, test_dict, test_list, test_list_dict):
            return paddle.less_than(i, ten)

        def body(i, ten, test_dict, test_list, test_list_dict):
            test_dict["test_key"] = i
            test_dict["test_key"] += 1

            test_list[0] = paddle.reshape(test_list[0], [2, -1]) + 1

            test_list_dict[0]["test_key"] += 1
            test_list_dict[0]["test_key"] = F.relu(
                test_list_dict[0]["test_key"]
            )

            i = paddle.increment(i)
            return [i, ten, test_dict, test_list, test_list_dict]

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.zeros(shape=[1], dtype='int64')
            ten = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=10
            )
            test_data = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=0
            )

            test_dict = {"test_key": test_data}
            test_list = [
                paddle.tensor.fill_constant(
                    shape=[2, 1], dtype='int64', value=0
                )
            ]
            test_list_dict = [
                {
                    "test_key": paddle.tensor.fill_constant(
                        shape=[1], dtype='float32', value=0
                    )
                }
            ]

            (
                i,
                ten,
                test_dict,
                test_list,
                test_list_dict,
            ) = paddle.static.nn.while_loop(
                cond, body, [i, ten, test_dict, test_list, test_list_dict]
            )
        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        res = exe.run(
            main_program,
            fetch_list=[
                test_dict["test_key"],
                test_list[0],
                test_list_dict[0]["test_key"],
            ],
        )
        np.testing.assert_allclose(
            np.asarray(res[0]),
            np.full(shape=1, fill_value=10, dtype=np.int64),
            rtol=1e-05,
        )
        np.testing.assert_allclose(
            np.asarray(res[1]),
            np.full(shape=(2, 1), fill_value=10, dtype=np.int64),
            rtol=1e-05,
        )
        np.testing.assert_allclose(
            np.asarray(res[2]),
            np.full(shape=1, fill_value=10, dtype=np.float32),
            rtol=1e-05,
        )


class TestApiWhileLoop_Nested(unittest.TestCase):

    @compare_legacy_with_pt
    def test_nested_net(self):
        def external_cond(i, j, init, sums):
            return paddle.less_than(i, loop_len1)

        def external_body(i, j, init, sums):
            def internal_cond(j, init, sums):
                return paddle.less_than(j, loop_len2)

            def internal_body(j, init, sums):
                init = paddle.add(x=init, y=ones)
                sums = paddle.add(x=init, y=sums)
                j = paddle.increment(j)
                return [j, init, sums]

            result = paddle.static.nn.while_loop(
                internal_cond, internal_body, [j, init, sums]
            )
            j = result[0]
            init = result[1]
            sums = result[2]
            sums = paddle.add(x=init, y=sums)
            i = paddle.increment(i)
            return [i, j, init, sums]

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.zeros(shape=[1], dtype='int64')
            j = paddle.zeros(shape=[1], dtype='int64')
            init = paddle.static.data(
                name='init', shape=[3, 3], dtype='float32'
            )
            sums = paddle.static.data(
                name='sums', shape=[3, 3], dtype='float32'
            )
            loop_len1 = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=2
            )
            loop_len2 = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=3
            )
            ones = paddle.tensor.fill_constant(
                shape=[3, 3], dtype='float32', value=1
            )

            out = paddle.static.nn.while_loop(
                external_cond, external_body, [i, j, init, sums]
            )

            data = np.random.rand(3, 3).astype('float32')
            data_sums = np.zeros([3, 3]).astype('float32')

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        res = exe.run(
            main_program, feed={'init': data, 'sums': data_sums}, fetch_list=out
        )
        for i in range(3):
            data = np.add(data, 1)
            data_sums = np.add(data, data_sums)
        for j in range(2):
            data_sums = np.add(data, data_sums)
        np.testing.assert_allclose(np.asarray(res[3]), data_sums, rtol=1e-05)


class TestApiWhileLoop_Backward(unittest.TestCase):
    def test_while_loop_backward(self):
        with paddle.pir_utils.IrGuard():

            def cond(i, x):
                return paddle.less_than(i, eleven)

            def body(i, x):
                x = paddle.multiply(x=i, y=i)
                i = paddle.increment(i)
                return [i, x]

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                i = paddle.static.data(name='i', shape=[1], dtype='float32')
                i.stop_gradient = False
                i.persistable = True
                eleven = paddle.tensor.fill_constant(
                    shape=[1], dtype='float32', value=11
                )
                one = paddle.tensor.fill_constant(
                    shape=[1], dtype='float32', value=1
                )
                x = paddle.static.data(name='x', shape=[1], dtype='float32')
                x.stop_gradient = False
                x.persistable = True

                out = paddle.static.nn.while_loop(cond, body, [i, x])
                mean = paddle.mean(out[1])
                grad_list = append_backward(mean)

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)

            feed_i = np.ones(1).astype('float32')
            feed_x = np.ones(1).astype('float32')
            data = np.asarray([100]).astype('float32')
            i_grad = np.asarray([20]).astype('float32')
            x_grad = np.asarray([0]).astype('float32')

            for p, g in grad_list:
                if p.is_same(i):
                    di = g
                elif p.is_same(x):
                    dx = g
            res = exe.run(
                main_program,
                feed={'i': feed_i, 'x': feed_x},
                fetch_list=[mean, di, dx],
            )

            np.testing.assert_allclose(np.asarray(res[0]), data, rtol=1e-05)
            np.testing.assert_allclose(np.asarray(res[1]), i_grad, rtol=1e-05)
            np.testing.assert_allclose(np.asarray(res[2]), x_grad, rtol=1e-05)

    def test_while_loop_backward2(self):
        def cond(i, x):
            return i < 3

        def body(i, x):
            x = x * i
            i = i + 1
            return [i, x]

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.static.data(name='i', shape=[1], dtype='float32')
            i.stop_gradient = False
            i.persistable = True
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False
            x.persistable = True

            out = paddle.static.nn.while_loop(cond, body, [i, x])
            mean = paddle.mean(out[1])
            grad_list = append_backward(mean)

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)

        feed_i = np.ones(1).astype('float32')
        feed_x = np.ones(1).astype('float32')
        data = np.asarray([2]).astype('float32')
        i_grad = np.asarray([3]).astype('float32')
        x_grad = np.asarray([2]).astype('float32')

        if paddle.framework.in_pir_mode():
            fetch_list = [out[1]]
            for p, g in grad_list:
                fetch_list.append(g)
            res = exe.run(
                main_program,
                feed={'i': feed_i, 'x': feed_x},
                fetch_list=fetch_list,
            )
        else:
            res = exe.run(
                main_program,
                feed={'i': feed_i, 'x': feed_x},
                fetch_list=[out[1].name, i.grad_name, x.grad_name],
            )

        np.testing.assert_allclose(np.asarray(res[0]), data, rtol=1e-05)
        np.testing.assert_allclose(np.asarray(res[1]), i_grad, rtol=1e-05)
        np.testing.assert_allclose(np.asarray(res[2]), x_grad, rtol=1e-05)


class TestApiWhileLoop_NestedWithBackwardAndDenseTensorArray(unittest.TestCase):
    # TODO(zhangbo): Support while grad exe for pir

    def test_nested_net_with_backward_and_lodtensor(self):
        def external_cond(i, j, x, mem_array):
            return paddle.less_than(i, array_len)

        def external_body(i, j, x, mem_array):
            def internal_cond(j, x, mem_array):
                return paddle.less_than(j, array_len2)

            def internal_body(j, x, mem_array):
                inner_data = paddle.tensor.array_read(array=data_array, i=j)
                inner_prev = paddle.tensor.array_read(array=mem_array, i=j)
                inner_sum_0 = paddle.add(x=inner_data, y=inner_prev)
                inner_sum_1 = paddle.add(x=x, y=inner_sum_0)
                j = paddle.increment(x=j)
                paddle.tensor.array_write(inner_sum_1, i=j, array=mem_array)

                return [j, x, mem_array]

            outer_data = paddle.tensor.array_read(array=data_array, i=i)
            outer_prev = paddle.tensor.array_read(array=mem_array, i=i)
            outer_sum_0 = paddle.add(x=outer_data, y=outer_prev)
            outer_sum_1 = paddle.add(x=x, y=outer_sum_0)
            i = paddle.increment(x=i)
            paddle.tensor.array_write(outer_sum_1, i=i, array=mem_array)
            j, x, mem_array = paddle.static.nn.while_loop(
                internal_cond, internal_body, [j, x, mem_array]
            )
            return [i, j, x, mem_array]

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            d0 = paddle.static.data(name='d0', shape=[10], dtype='float32')
            d1 = paddle.static.data(name='d1', shape=[10], dtype='float32')
            d2 = paddle.static.data(name='d2', shape=[10], dtype='float32')
            x = paddle.static.data(name='x', shape=[10], dtype='float32')
            d0.persistable = True
            d0.stop_gradient = False
            d1.persistable = True
            d2.persistable = True
            x.stop_gradient = False
            x.persistable = True
            i = paddle.zeros(shape=[1], dtype='int64')
            i.stop_gradient = True
            i.persistable = True
            init = paddle.zeros(shape=[10], dtype='float32')
            init.stop_gradient = False
            mem_array = paddle.tensor.array_write(x=init, i=i)
            data_array = paddle.tensor.array_write(x=d0, i=i)
            mem_array.stop_gradient = False
            data_array.stop_gradient = False
            mem_array.persistable = True
            i = paddle.increment(i)
            paddle.tensor.array_write(d1, i, array=data_array)
            i = paddle.increment(i)
            paddle.tensor.array_write(d2, i, array=data_array)
            i = paddle.zeros(shape=[1], dtype='int64')
            i.stop_gradient = True
            array_len = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=1
            )
            j = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
            j.stop_gradient = True
            array_len2 = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=3
            )

            out = paddle.static.nn.while_loop(
                external_cond, external_body, [i, j, x, mem_array]
            )

            sum_result = paddle.tensor.array_read(array=out[3], i=j)
            mean = paddle.mean(sum_result)
            grad_list = append_backward(mean)
            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)

            d = []
            for i in range(3):
                d.append(np.ones(10).astype('float32'))
            feed_x = np.ones(10).astype('float32')
            data_sum = d[0] + d[1] + d[2] + 3 * feed_x
            x_grad = [0.3] * 10
            if paddle.framework.in_pir_mode():
                for p, g in grad_list:
                    if p.is_same(x):
                        dx = g
                res = exe.run(
                    main_program,
                    feed={'d0': d[0], 'd1': d[1], 'd2': d[2], 'x': feed_x},
                    fetch_list=[sum_result, dx],
                )
            else:
                res = exe.run(
                    main_program,
                    feed={'d0': d[0], 'd1': d[1], 'd2': d[2], 'x': feed_x},
                    fetch_list=[sum_result.name, x.grad_name],
                )
            np.testing.assert_allclose(res[0], data_sum, rtol=1e-05)
            np.testing.assert_allclose(res[1], x_grad, rtol=1e-05)

    def test_while_backward_with_inplace(self):
        with paddle.pir_utils.IrGuard():

            def internal_cond(i, x, mem_array):
                return paddle.less_than(i, array_len)

            def internal_body(i, x, mem_array):
                t0 = paddle.tensor.array_read(array=mem_array, i=i)
                t1 = paddle.add(t0, x)
                i = paddle.increment(i)
                paddle.tensor.array_write(t1, i, array=mem_array)
                return [i, x, mem_array]

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                i = paddle.zeros(shape=[1], dtype='int64')
                x = paddle.static.data(name='x', shape=[10], dtype='float32')
                x.stop_gradient = False
                init = paddle.zeros(shape=[10], dtype='float32')
                mem_array = paddle.tensor.array_write(init, i)
                mem_array.stop_gradient = False
                array_len = paddle.tensor.fill_constant(
                    shape=[1], dtype='int64', value=3
                )

                i, x, mem_array = paddle.static.nn.while_loop(
                    internal_cond, internal_body, [i, x, mem_array]
                )

                out = paddle.tensor.array_read(mem_array, i)
                mean_out = paddle.mean(out)
                dx, dmem_array = paddle.static.gradients(
                    mean_out, [x, mem_array]
                )

                j = paddle.zeros(shape=[1], dtype='int64')
                dmem0 = paddle.tensor.array_read(dmem_array, j)
                j = paddle.increment(j)
                dmem1 = paddle.tensor.array_read(dmem_array, j)
                j = paddle.increment(j)
                dmem2 = paddle.tensor.array_read(dmem_array, j)
                j = paddle.increment(j)
                dmem3 = paddle.tensor.array_read(dmem_array, j)
                place = (
                    base.CUDAPlace(0)
                    if core.is_compiled_with_cuda()
                    else base.CPUPlace()
                )
                exe = base.Executor(place)

                feed_x = np.ones(10).astype('float32')

                res = exe.run(
                    main_program,
                    feed={"x": feed_x},
                    fetch_list=[out, dx, dmem0, dmem1, dmem2, dmem3],
                )

                np.testing.assert_allclose(res[0], [3] * 10, rtol=1e-05)
                np.testing.assert_allclose(res[1], [0.3] * 10, rtol=1e-05)
                np.testing.assert_allclose(res[2], [0.0] * 10, rtol=1e-05)
                np.testing.assert_allclose(res[3], [0.0] * 10, rtol=1e-05)
                np.testing.assert_allclose(res[4], [0.0] * 10, rtol=1e-05)
                np.testing.assert_allclose(res[5], [0.0] * 10, rtol=1e-05)


class TestApiWhileLoopWithSwitchCase(unittest.TestCase):
    @compare_legacy_with_pt
    def test_with_switch_case(self):
        def cond(i):
            return paddle.less_than(i, ten)

        def body(i):
            def fn_add_three():
                data_add_three = paddle.add(x=i, y=three)
                return data_add_three

            def fn_square():
                data_mul_data = paddle.multiply(x=i, y=i)
                return data_mul_data

            def fn_add_one():
                data_add_one = paddle.add(x=i, y=one)
                return data_add_one

            return paddle.static.nn.switch_case(
                branch_index=i,
                branch_fns={2: fn_add_three, 5: fn_square},
                default=fn_add_one,
            )

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
            ten = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=10
            )
            three = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=3
            )
            one = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
            out = paddle.static.nn.while_loop(cond, body, [i])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        res = exe.run(main_program, fetch_list=out)

        data = np.asarray([25]).astype('int64')
        np.testing.assert_allclose(np.asarray(res[0]), data, rtol=1e-05)


class TestApiWhileLoop_Error(unittest.TestCase):
    @compare_legacy_with_pt
    def test_error1(self):
        def cond_returns_constant(i):
            return 1

        def cond_returns_not_bool_tensor(i):
            return paddle.increment(i)

        def cond_returns_bool_tensor(i):
            return paddle.less_than(i, ten)

        def cond_returns_2d_tensor(i):
            return paddle.less_than(i, ten_2d)

        def cond_receives_two_args(i, ten):
            return paddle.less_than(i, ten)

        def body(i):
            return paddle.increment(i)

        def body_returns_error_length(i):
            i = paddle.increment(i)
            return [i, i]

        def body_returns_error_type(i, ten):
            return paddle.increment(i)

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=1
            )
            data_1d = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=1
            )
            data_2d = paddle.tensor.fill_constant(
                shape=[2, 2], dtype='int64', value=1
            )
            ten = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=10
            )
            ten_2d = paddle.tensor.fill_constant(
                shape=[2, 2], dtype='int64', value=10
            )

            # The type of `cond` in Op(while_loop) must be callable
            def type_error_cond():
                out = paddle.static.nn.while_loop(data, body, [data_1d])

            self.assertRaises(TypeError, type_error_cond)

            # The type of `body` in Op(while_loop) must be callable
            def type_error_body():
                out = paddle.static.nn.while_loop(
                    cond_returns_bool_tensor, data, [data_1d]
                )

            self.assertRaises(TypeError, type_error_body)

            # The type of `loop_vars` in Op(while_loop) must be list or tuple
            def type_error_loop_vars():
                out = paddle.static.nn.while_loop(
                    cond_returns_bool_tensor, body, data_1d
                )

            self.assertRaises(TypeError, type_error_loop_vars)

            # The value of `loop_vars` is empty
            def value_error_loop_vars():
                out = paddle.static.nn.while_loop(
                    cond_returns_bool_tensor, body, []
                )

            self.assertRaises(ValueError, value_error_loop_vars)

            # The type of `cond` returns in Op(while_loop) must be Variable
            def type_error_cond_returns_not_variable():
                out = paddle.static.nn.while_loop(
                    cond_returns_constant, body, [data_1d]
                )

            self.assertRaises(TypeError, type_error_cond_returns_not_variable)

            # The type of `cond` returns in Op(while_loop) must be a boolean variable
            def type_error_cond_returns_not_boolean():
                out = paddle.static.nn.while_loop(
                    cond_returns_not_bool_tensor, body, [data_1d]
                )

            self.assertRaises(TypeError, type_error_cond_returns_not_boolean)

            # The shape of `cond` returns in Op(while_loop) must be 1
            def type_error_shape_cond_returns_2d():
                out = paddle.static.nn.while_loop(
                    cond_returns_2d_tensor, body, [data_2d]
                )

            self.assertRaises(TypeError, type_error_shape_cond_returns_2d)

            # The length of `body` returns in Op(while_loop) must be same as `loop_vars`
            def value_error_body_returns_error_length():
                out = paddle.static.nn.while_loop(
                    cond_returns_bool_tensor, body_returns_error_length, [data]
                )

            self.assertRaises(ValueError, value_error_body_returns_error_length)

            # The type of `body` returns in Op(while_loop) must be same as `loop_vars`
            def value_error_body_returns_error_type():
                out = paddle.static.nn.while_loop(
                    cond_receives_two_args, body_returns_error_type, [data, ten]
                )

            self.assertRaises(ValueError, value_error_body_returns_error_type)

    @compare_legacy_with_pt
    def test_error2(self):
        def cond_returns_with_mutable_dict(i, test_dict):
            return i > 0

        def body_returns_with_mutable_dict(i, test_dict):
            test_dict['new_key'] = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=1
            )
            return paddle.increment(i), test_dict

        def cond_returns_with_mutable_list(i, test_list):
            return i > 0

        def body_returns_with_mutable_list(i, test_list):
            test_list.append(
                paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
            )
            return paddle.increment(i), test_list

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=1
            )

            # The length of `output_vars` with mutable value should keep same with `loop_vars`
            # TODO(zhangbo): slice error need to fix, loop_vars support list/dict
            def value_error_body_returns_with_mutable_dict():
                test_dict = {
                    "int_constant": paddle.tensor.fill_constant(
                        shape=[2, 2], dtype='int64', value=1
                    )
                }
                out = paddle.static.nn.while_loop(
                    cond_returns_with_mutable_dict,
                    body_returns_with_mutable_dict,
                    [data, test_dict],
                )

            self.assertRaises(
                ValueError, value_error_body_returns_with_mutable_dict
            )

            # TODO(zhangbo): loop_vars support list/dict
            def value_error_body_returns_with_mutable_list():
                test_list = [
                    paddle.tensor.fill_constant(
                        shape=[2, 2], dtype='int64', value=1
                    )
                ]
                out = paddle.static.nn.while_loop(
                    cond_returns_with_mutable_list,
                    body_returns_with_mutable_list,
                    [data, test_list],
                )

            self.assertRaises(
                ValueError, value_error_body_returns_with_mutable_list
            )


class TestApiWhileLoopSliceInBody(unittest.TestCase):
    @compare_legacy_with_pt
    def test_var_slice(self):
        def cond(z, i):
            return i + 1 <= x_shape[0]

        def body(z, i):
            z = z + x[i]
            i += 1
            return z, i

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name='x', shape=[-1, 5], dtype='int32')
            z = paddle.tensor.fill_constant([], 'int32', 0)
            x_shape = paddle.shape(x)
            i = paddle.tensor.fill_constant([], 'int32', 0)
            z, _ = paddle.static.nn.while_loop(cond, body, [z, i])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)

        np_x = np.array([1, 2, 3, 4, 5], dtype='int32')
        res = exe.run(main_program, feed={'x': np_x}, fetch_list=[z])
        np.testing.assert_array_equal(res[0], [np.sum(np_x)])


if __name__ == '__main__':
    unittest.main()
