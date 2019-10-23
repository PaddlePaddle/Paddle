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

import unittest

import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.framework import default_startup_program


class TestCond(unittest.TestCase):
    def run_cond(self,
                 x_value,
                 with_if=True,
                 with_elif=True,
                 with_else=True,
                 if_after_else=False,
                 elif_after_else=False,
                 else_after_else=False):
        x = layers.fill_constant(shape=[1], dtype='float32', value=x_value)

        zero_var = layers.fill_constant(shape=[1], dtype='float32', value=0.0)
        one_var = layers.fill_constant(shape=[1], dtype='float32', value=1.0)
        two_var = layers.fill_constant(shape=[1], dtype='float32', value=2.0)
        three_var = layers.fill_constant(shape=[1], dtype='float32', value=3.0)

        result = layers.fill_constant(shape=[1], value=-1.0, dtype='float32')

        with layers.Cond() as cond:
            if with_if:
                with cond.if_case(layers.less_than(x, zero_var)):
                    layers.assign(zero_var, result)
            if with_elif:
                with cond.elif_case(layers.less_than(x, one_var)):
                    layers.assign(one_var, result)
                with cond.elif_case(layers.less_than(x, two_var)):
                    layers.assign(two_var, result)
            if with_else:
                with cond.else_case():
                    layers.assign(three_var, result)

            # These cases are called after else, so it may report ValueError
            if if_after_else:
                with cond.if_case(layers.less_than(x, zero_var)):
                    layers.assign(zero_var, result)
            if elif_after_else:
                with cond.elif_case(layers.less_than(x, one_var)):
                    layers.assign(one_var, result)
                with cond.elif_case(layers.less_than(x, two_var)):
                    layers.assign(two_var, result)
            if else_after_else:
                with cond.else_case():
                    layers.assign(three_var, result)

        cpu = core.CPUPlace()
        exe = Executor(cpu)
        exe.run(default_startup_program())

        out = exe.run(feed={}, fetch_list=[result])
        return out[0][0]

    def test_if_elif_else(self):
        test_data = [(-0.1, 0), (0.1, 1), (1.0, 2), (2.1, 3)]
        for x, expected_result in test_data:
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                result = self.run_cond(x)
                self.assertEqual(result, expected_result)

    def test_if_else(self):
        test_data = [(-0.1, 0), (0.0, 3), (0.1, 3), (1, 3)]
        for x, expected_result in test_data:
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                result = self.run_cond(x, with_elif=False)
                self.assertEqual(result, expected_result)

    def test_if(self):
        test_data = [(-0.1, 0), (0.0, -1), (0.1, -1), (1, -1)]
        for x, expected_result in test_data:
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                result = self.run_cond(x, with_elif=False, with_else=False)
                self.assertEqual(result, expected_result)

    def test_if_elif(self):
        test_data = [(-0.1, 0), (0.1, 1), (1.0, 2), (2.1, -1)]
        for x, expected_result in test_data:
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                result = self.run_cond(x, with_else=False)
                self.assertEqual(result, expected_result)

    def test_no_if_call_elif(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            with self.assertRaises(ValueError) as err:
                result = self.run_cond(0, with_if=False)
            self.assertEqual("You should call if_case before elif_case",
                             str(err.exception))

    def test_no_if_call_else(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            with self.assertRaises(ValueError) as err:
                result = self.run_cond(0, with_if=False, with_elif=False)
            self.assertEqual("You should call if_case before else_case",
                             str(err.exception))

    def test_if_after_else(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            with self.assertRaises(ValueError) as err:
                result = self.run_cond(0, if_after_else=True)
            self.assertEqual("You can not call if_case after else_case",
                             str(err.exception))

    def test_elif_after_else(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            with self.assertRaises(ValueError) as err:
                result = self.run_cond(0, elif_after_else=True)
            self.assertEqual("You can not call elif_case after else_case",
                             str(err.exception))

    def test_else_after_else(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            with self.assertRaises(ValueError) as err:
                result = self.run_cond(0, else_after_else=True)
            self.assertEqual("You can not call else_case after else_case",
                             str(err.exception))

    def test_call_out_of_with(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=1.0)
            result = layers.fill_constant(
                shape=[1], value=-1.0, dtype='float32')
            with self.assertRaises(ValueError) as err:
                cond = layers.Cond()
                cond.if_case(layers.less_than(result, x))
                layers.assign(x, result)
            self.assertEqual(
                "if_case should be called in 'with layers.Cond() as ... :'",
                str(err.exception))

            with layers.Cond() as cond:
                cond.if_case(layers.less_than(result, x))
                layers.assign(x, result)

            cpu = core.CPUPlace()
            exe = Executor(cpu)
            exe.run(default_startup_program())
            out = exe.run(feed={}, fetch_list=[result])
            self.assertEqual(out[0][0], 1.0)


if __name__ == '__main__':
    unittest.main()
