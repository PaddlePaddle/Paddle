#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

paddle.enable_static()


class TestGetInputsOutputsInBlock(unittest.TestCase):
    def test_ordered(self):
        # Program variable names may be different when test order is different
        # This helper makes the test ordered.
        self._test_while_loop()
        self._test_cond()

    def _test_while_loop(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.assign(np.array([1]))
            ten = paddle.assign(np.array([10]))

            def while_cond(i):
                # use ten in parent block without passing it
                return i < ten

            def while_body(i):
                # variable created in sub block
                one = paddle.assign(np.array([1]))
                i = i + one
                return [i]

            i = paddle.static.nn.while_loop(while_cond, while_body, [i])

        sub_block = main_program.block(1)
        (
            inner_inputs,
            inner_outputs,
        ) = paddle.utils.get_inputs_outputs_in_block(sub_block)
        # 'assign_0.tmp_0', 'assign_1.tmp_0' are name of i and ten in program
        self.assertTrue(inner_inputs == {'assign_0.tmp_0', 'assign_1.tmp_0'})
        # 'tmp_0', 'assign_0.tmp_0' are name of i < ten and i in program
        self.assertTrue(inner_outputs == {'tmp_0', 'assign_0.tmp_0'})

    def _test_cond(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.zeros((1, 1))
            b = paddle.zeros((1, 1))
            c = a * b
            out = paddle.static.nn.cond(a < b, lambda: a + c, lambda: b * b)

        sub_block = main_program.block(1)
        (
            inner_inputs,
            inner_outputs,
        ) = paddle.utils.get_inputs_outputs_in_block(sub_block)
        # 'fill_constant_1.tmp_0', 'tmp_3' are names of a, c
        self.assertTrue(inner_inputs == {'fill_constant_1.tmp_0', 'tmp_0'})
        # '_generated_var_1', is name of a + c
        self.assertTrue(inner_outputs == {'_generated_var_0'})


if __name__ == "__main__":
    unittest.main()
