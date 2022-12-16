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

import unittest
from functools import partial

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard

paddle.enable_static()


class TestAPISwitchCase(unittest.TestCase):
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
            index_1 = layers.fill_constant(shape=[1], dtype='int32', value=1)
            index_2 = layers.fill_constant(shape=[1], dtype='int32', value=2)
            index_5 = layers.fill_constant(shape=[1], dtype='int32', value=5)

            # call fn_1
            out_0 = paddle.static.nn.switch_case(
                branch_index=index_1, branch_fns={1: fn_1, 2: fn_2, 3: fn_3}
            )

            # call fn_2 : branch_fns={0: fn_1, 1:fn_2, 2:fn_3}
            out_1 = paddle.static.nn.switch_case(
                branch_index=index_1, branch_fns=(fn_1, fn_2, fn_3)
            )

            # call default fn_3
            out_2 = paddle.static.nn.switch_case(
                branch_index=index_5,
                branch_fns=((1, fn_1), (2, fn_2)),
                default=fn_3,
            )

            # no default, call fn_2
            out_3 = paddle.static.nn.switch_case(
                branch_index=index_2, branch_fns=[(1, fn_1), (2, fn_2)]
            )

            # no default, call fn_2 but branch_index is 5
            out_4 = paddle.static.nn.switch_case(
                branch_index=index_5,
                branch_fns=[(1, fn_1), (3, fn_2), (2, fn_3)],
            )

            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
            exe = fluid.Executor(place)

            res = exe.run(
                main_program, fetch_list=[out_0, out_1, out_2, out_3, out_4]
            )

            np.testing.assert_allclose(
                res[0],
                1,
                rtol=1e-05,
                err_msg='result is {} but answer is {}'.format(res[0], 1),
            )
            np.testing.assert_allclose(
                res[1],
                2,
                rtol=1e-05,
                err_msg='result is {} but answer is {}'.format(res[0], 2),
            )
            np.testing.assert_allclose(
                res[2],
                3,
                rtol=1e-05,
                err_msg='result is {} but answer is {}'.format(res[0], 3),
            )
            np.testing.assert_allclose(
                res[3],
                2,
                rtol=1e-05,
                err_msg='result is {} but answer is {}'.format(res[0], 2),
            )
            np.testing.assert_allclose(
                res[4],
                2,
                rtol=1e-05,
                err_msg='result is {} but answer is {}'.format(res[0], 2),
            )

    def test_return_var_tuple(self):
        def fn_1():
            return layers.fill_constant(
                shape=[1, 2], dtype='int32', value=1
            ), layers.fill_constant(shape=[2, 3], dtype='float32', value=2)

        def fn_2():
            return layers.fill_constant(
                shape=[3, 4], dtype='int32', value=3
            ), layers.fill_constant(shape=[4, 5], dtype='float32', value=4)

        def fn_3():
            return layers.fill_constant(
                shape=[5], dtype='int32', value=5
            ), layers.fill_constant(shape=[5, 6], dtype='float32', value=6)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            index_1 = layers.fill_constant(shape=[1], dtype='int32', value=1)

            out = paddle.static.nn.switch_case(
                index_1, ((1, fn_1), (2, fn_2)), fn_3
            )

            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
            exe = fluid.Executor(place)
            ret = exe.run(main_program, fetch_list=out)

            np.testing.assert_allclose(
                np.asarray(ret[0]), np.full((1, 2), 1, np.int32), rtol=1e-05
            )
            np.testing.assert_allclose(
                np.asarray(ret[1]), np.full((2, 3), 2, np.float32), rtol=1e-05
            )


class TestAPISwitchCase_Nested(unittest.TestCase):
    def test_nested_switch_case(self):
        def fn_1(x=1):
            out = paddle.static.nn.switch_case(
                branch_index=layers.fill_constant(
                    shape=[1], dtype='int32', value=x
                ),
                branch_fns={
                    1: partial(
                        layers.fill_constant, shape=[1], dtype='int32', value=1
                    ),
                    x: partial(
                        layers.fill_constant, shape=[2], dtype='int32', value=x
                    ),
                },
            )
            return out

        def fn_2(x=2):
            out = paddle.static.nn.switch_case(
                branch_index=layers.fill_constant(
                    shape=[1], dtype='int32', value=2
                ),
                branch_fns={
                    1: partial(
                        layers.fill_constant,
                        shape=[4, 3],
                        dtype='int32',
                        value=1,
                    ),
                    2: partial(fn_1, x=x),
                },
            )
            return out

        def fn_3():
            out = paddle.static.nn.switch_case(
                branch_index=layers.fill_constant(
                    shape=[1], dtype='int32', value=3
                ),
                branch_fns={
                    1: partial(
                        layers.fill_constant,
                        shape=[4, 3],
                        dtype='int32',
                        value=1,
                    ),
                    3: partial(fn_2, x=3),
                },
            )
            return out

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            index_1 = fluid.data(name="index_1", shape=[1], dtype='uint8')
            index_2 = layers.fill_constant(shape=[1], dtype='int32', value=2)
            index_3 = layers.fill_constant(shape=[1], dtype='int64', value=3)

            out_1 = paddle.static.nn.switch_case(
                branch_index=index_1, branch_fns={1: fn_1, 2: fn_2, 3: fn_3}
            )
            out_2 = paddle.static.nn.switch_case(
                branch_index=index_2, branch_fns={1: fn_1, 2: fn_2, 3: fn_3}
            )

            out_3 = paddle.static.nn.switch_case(
                branch_index=index_3, branch_fns={1: fn_1, 2: fn_2, 3: fn_3}
            )

            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
            exe = fluid.Executor(place)

            res = exe.run(
                main_program,
                feed={"index_1": np.array([1], dtype="uint8")},
                fetch_list=[out_1, out_2, out_3],
            )

            np.testing.assert_allclose(
                res[0],
                1,
                rtol=1e-05,
                err_msg='result is {} but answer is {}'.format(res[0], 1),
            )
            np.testing.assert_allclose(
                res[1],
                2,
                rtol=1e-05,
                err_msg='result is {} but answer is {}'.format(res[1], 2),
            )
            np.testing.assert_allclose(
                res[2],
                3,
                rtol=1e-05,
                err_msg='result is {} but answer is {}'.format(res[2], 3),
            )


# test TypeError and ValueError of api switch_case
class TestAPISwitchCase_Error(unittest.TestCase):
    def test_error(self):
        def fn_1():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=1)

        def fn_2():
            return layers.fill_constant(shape=[4, 2], dtype='int32', value=2)

        def fn_3():
            return layers.fill_constant(shape=[4, 3], dtype='int32', value=3)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            key_float32 = layers.fill_constant(
                shape=[1], dtype='float32', value=0.23
            )
            key_int32 = layers.fill_constant(
                shape=[1], dtype='int32', value=0.23
            )

            # The type of 'branch_index' in Op(switch_case) must be Variable
            def type_error_branch_index():
                paddle.static.nn.switch_case(
                    branch_index=1, branch_fns=[(1, fn_1)], default=fn_3
                )

            self.assertRaises(TypeError, type_error_branch_index)

            # The data type of 'branch_index' in Op(switch_case) must be int32, int64 or uint8
            def dtype_error_branch_index():
                paddle.static.nn.switch_case(
                    branch_index=key_float32,
                    branch_fns=[(1, fn_1)],
                    default=fn_3,
                )

            self.assertRaises(TypeError, dtype_error_branch_index)

            # The type of 'branch_fns' in Op(switch_case) must be list, tuple or dict
            def type_error_branch_fns():
                paddle.static.nn.switch_case(
                    branch_index=key_int32, branch_fns=1, default=fn_3
                )

            self.assertRaises(TypeError, type_error_branch_fns)

            # The elements' type of 'branch_fns' in Op(switch_case) must be tuple
            def type_error_index_fn_pair_1():
                paddle.static.nn.switch_case(
                    branch_index=key_int32, branch_fns=[1], default=fn_3
                )

            self.assertRaises(TypeError, type_error_index_fn_pair_1)

            # The tuple's size of 'branch_fns' in Op(switch_case) must be 2
            def type_error_index_fn_pair_2():
                paddle.static.nn.switch_case(
                    branch_index=key_int32, branch_fns=[(1, 2, 3)], default=fn_3
                )

            self.assertRaises(TypeError, type_error_index_fn_pair_2)

            # The key's type of 'branch_fns' in Op(switch_case) must be int
            def type_error_key():
                paddle.static.nn.switch_case(
                    branch_index=key_int32, branch_fns=[(2.3, 2)], default=fn_3
                )

            self.assertRaises(TypeError, type_error_key)

            # The key in 'branch_fns' must be unique
            def value_error_key():
                paddle.static.nn.switch_case(
                    branch_index=key_int32,
                    branch_fns=[(2, fn_1), (2, fn_2)],
                    default=fn_3,
                )

            self.assertRaises(ValueError, value_error_key)

            # The type of function in 'branch_fns' must be callable
            def type_error_fn():
                paddle.static.nn.switch_case(
                    branch_index=key_int32,
                    branch_fns=[(1, 1), (2, fn_2)],
                    default=fn_3,
                )

            self.assertRaises(TypeError, type_error_fn)

            # The default in Op(case) must be callable
            def type_error_default():
                paddle.static.nn.switch_case(
                    branch_index=key_int32,
                    branch_fns=[(1, fn_1), (2, fn_2)],
                    default=1,
                )

            self.assertRaises(TypeError, type_error_default)


if __name__ == '__main__':
    unittest.main()
