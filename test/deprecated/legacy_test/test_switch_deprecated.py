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

import paddle
from paddle.base import core, framework
from paddle.base.executor import Executor
from paddle.base.framework import default_startup_program

paddle.enable_static()


class TestSwitch(unittest.TestCase):
    def check_switch(self, value):
        x = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=value)
        zero_var = paddle.tensor.fill_constant(
            shape=[1], dtype='float32', value=0.0
        )
        one_var = paddle.tensor.fill_constant(
            shape=[1], dtype='float32', value=1.0
        )
        two_var = paddle.tensor.fill_constant(
            shape=[1], dtype='float32', value=2.0
        )
        three_var = paddle.tensor.fill_constant(
            shape=[1], dtype='float32', value=3.0
        )

        result = paddle.static.create_global_var(
            shape=[1], value=-1.0, dtype='float32', persistable=True
        )

        res = paddle.static.nn.case(
            pred_fn_pairs=[
                (paddle.less_than(x, zero_var), lambda: zero_var),
                (paddle.less_than(x, one_var), lambda: one_var),
                (paddle.less_than(x, two_var), lambda: two_var),
            ],
            default=lambda: three_var,
        )
        paddle.assign(res, result)

        cpu = core.CPUPlace()
        exe = Executor(cpu)
        exe.run(default_startup_program())

        out = exe.run(feed={}, fetch_list=[result])[0][0]
        return out

    def test_switch(self):
        test_data = {(-0.1, 0), (0.1, 1), (1.1, 2), (2.1, 3)}
        for x, expected_result in test_data:
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                result = self.check_switch(x)
                self.assertEqual(result, expected_result)


class TestSwitchCaseError(unittest.TestCase):
    def test_error(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            cond = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.0
            )
            zero_var = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.0
            )

            result = paddle.static.create_global_var(
                shape=[1], value=-1.0, dtype='float32', persistable=True
            )

            # 1. The type of 'condition' in case must be Variable.
            def test_condition_type():
                res = paddle.static.nn.case(
                    [(1, lambda: zero_var)], default=lambda: result
                )
                paddle.assign(res, result)

            self.assertRaises(TypeError, test_condition_type)

            # 2. The dtype of 'condition' in case must be 'bool'.
            def test_condition_dtype():
                res = paddle.static.nn.case(
                    [cond, lambda: zero_var], default=lambda: result
                )
                paddle.assign(res, result)

            self.assertRaises(TypeError, test_condition_dtype)


if __name__ == '__main__':
    unittest.main()
