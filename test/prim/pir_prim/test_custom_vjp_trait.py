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
from paddle import nn, pir
from paddle.base.core import has_custom_vjp

paddle.enable_static()


def get_gelu_program_pir():
    with paddle.pir_utils.OldIrGuard():
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.static.data('x', [2, 3, 3], dtype='float32')
            net = nn.GELU()
            out = net(x)
        pir_program = pir.translate_to_pir(main_program.desc)
        return pir_program


def get_multiply_program_pir():
    with paddle.pir_utils.OldIrGuard():
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.static.data('x', [2, 3, 3], dtype='float32')
            y = paddle.static.data('y', [2, 3, 3], dtype='float32')
            out = paddle.multiply(x, y)
        pir_program = pir.translate_to_pir(main_program.desc)
        return pir_program


class TestCustomVjpTrait(unittest.TestCase):
    def test_gelu_op_custom_vjp_trait(self):
        pir_program = get_gelu_program_pir()
        op = pir_program.global_block().ops[-1]
        self.assertEqual(op.name(), "pd_op.gelu")
        self.assertEqual(has_custom_vjp(op), True)

    def test_multiply_op_custom_vjp_trait(self):
        pir_program = get_multiply_program_pir()
        op = pir_program.global_block().ops[-1]
        self.assertEqual(op.name(), "pd_op.multiply")
        self.assertEqual(has_custom_vjp(op), False)


if __name__ == "__main__":
    unittest.main()
