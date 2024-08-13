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
from paddle import pir
from paddle.base import core
from paddle.framework import LayerHelper

paddle.enable_static()


class TestShadowOutputSlice(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())
            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x = paddle.ones([3, 9, 5], dtype='float32')
                    y = paddle.static.data(
                        name="y", shape=[3, 9, 5], dtype="float32"
                    )
                    z = x * y  # will be eliminated

                    _, out, _ = paddle.split(x, num_or_sections=3, axis=1)
                    helper = LayerHelper('shadow_output')
                    helper.append_op(
                        type="shadow_output",
                        inputs={"x": [out.name]},
                        outputs={"out": [y.name]},
                        attrs={"name": out.name},
                    )

            new_program = pir.translate_to_pir(main_program.desc)
            op_names = [op.name() for op in new_program.global_block().ops]
            self.assertTrue('pd_op.multiply' in op_names)
            pm = pir.PassManager()
            pm.add_pass(
                'dead_code_elimination_pass', {}
            )  # apply pass to eliminate dead code
            pm.run(new_program)
            op_names = [op.name() for op in new_program.global_block().ops]
            self.assertEqual(pm.passes(), ['dead_code_elimination_pass'])
            self.assertFalse(pm.empty())
            self.assertTrue(
                'pd_op.multiply' not in op_names
            )  # multiply is eliminated because its output is not used


if __name__ == "__main__":
    unittest.main()
