# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import ir
from paddle.fluid import core

paddle.enable_static()


class TestPdInplacePass(unittest.TestCase):
    def test_pd_inplace_pass(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones([3, 9, 5], dtype='float32')
                y = paddle.nn.functional.relu(x)

        new_program = ir.translate_to_new_ir(main_program.desc)
        op_names = [op.name() for op in new_program.block().ops]
        print(op_names)
        print(new_program)
        self.assertTrue('pd.relu' in op_names)
        pm = ir.PassManager()
        pm.add_pass('InplacePass')  # apply pass to elimitate dead code
        pm.run(new_program)
        op_names = [op.name() for op in new_program.block().ops]
        print(op_names)
        print(new_program)
        # self.assertEqual(pm.passes(), ['DeadCodeEliminationPass'])
        # self.assertFalse(pm.empty())
        # self.assertTrue(
        #     'pd.uniform' not in op_names
        # )  # uniform is elimited because its output is not used


if __name__ == "__main__":
    unittest.main()
