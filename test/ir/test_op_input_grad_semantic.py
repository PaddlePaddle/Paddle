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

paddle.enable_static()


def get_gather_program_pir():
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.tensor.fill_constant(
            shape=[3, 4], dtype='float32', value=2.0
        )
        index = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=1.0)
        axis = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=2.0)
        out = paddle.gather(x, index, axis)
    pir_program = pir.translate_to_pir(main_program.desc)
    return pir_program


def get_multiply_program_pir():
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.tensor.fill_constant(
            shape=[3, 4], dtype='float32', value=2.0
        )
        y = paddle.tensor.fill_constant(
            shape=[3, 4], dtype='float32', value=3.0
        )
        out = paddle.multiply(x, y)
    pir_program = pir.translate_to_pir(main_program.desc)
    return pir_program


class TestOpInputGradSemantic(unittest.TestCase):
    def test_gather_op_input_grad_semantic(self):
        pir_program = get_gather_program_pir()
        gather_op = pir_program.global_block().ops[-1]
        self.assertEqual(
            gather_op.get_input_grad_semantics(), [True, False, False]
        )

    def test_multiply_op_input_grad_semantic(self):
        pir_program = get_multiply_program_pir()
        multiply_op = pir_program.global_block().ops[-1]
        self.assertEqual(multiply_op.get_input_grad_semantics(), [True, True])


if __name__ == "__main__":
    unittest.main()
