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
from paddle.base.core import call_vjp

paddle.enable_static()


def get_ir_divide_program():
    paddle.enable_static()
    with paddle.pir_utils.OldIrGuard():
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.tensor.fill_constant(
                shape=[1, 4], dtype='float32', value=2.0
            )
            x.stop_gradient = False
            y = paddle.tensor.fill_constant(
                shape=[4], dtype='float32', value=1.0
            )
            y.stop_gradient = False
            dout = paddle.tensor.fill_constant(
                shape=[1, 4], dtype='float32', value=1.0
            )
            dout.stop_gradient = False
            out = paddle.divide(x, y)
        pir_program = pir.translate_to_pir(main_program.desc)
        return pir_program


def get_ir_sum_program():
    paddle.enable_static()
    with paddle.pir_utils.OldIrGuard():
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.tensor.fill_constant(
                shape=[4, 5], dtype='float32', value=2.0
            )
            x.stop_gradient = False
            dout = paddle.tensor.fill_constant(
                shape=[], dtype='float32', value=1.0
            )
            dout.stop_gradient = False
            out = paddle.sum(x)
        pir_program = pir.translate_to_pir(main_program.desc)
        return pir_program


class TestVjpPrim(unittest.TestCase):
    def test_divide_grad_prim_case1(self):
        pir_program = get_ir_divide_program()
        paddle.framework.core._set_prim_backward_enabled(True)
        with paddle.pir_utils.IrGuard():
            dout = pir_program.global_block().ops[-2].result(0)
            out_grads = [[dout]]
            stop_gradients = [[False], [False]]
            divide_op = pir_program.global_block().ops[-1]
            with paddle.pir.core.program_guard(pir_program):
                grad_outs = call_vjp(
                    divide_op,
                    [[value] for value in divide_op.operands_source()],
                    [[value] for value in divide_op.results()],
                    out_grads,
                    stop_gradients,
                )
            print(pir_program)
            reshape_op2 = pir_program.global_block().ops[-1]
            reshape_op1 = pir_program.global_block().ops[-2]
            self.assertEqual(len(grad_outs), 2)
            self.assertEqual(len(pir_program.global_block().ops), 11)
            self.assertTrue(reshape_op2.result(0).is_same(grad_outs[0][0]))
            self.assertTrue(reshape_op1.result(0).is_same(grad_outs[1][0]))
            paddle.framework.core._set_prim_backward_enabled(False)

    def test_divide_grad_no_prim(self):
        pir_program = get_ir_divide_program()
        paddle.framework.core._set_prim_backward_enabled(False)
        dout = pir_program.global_block().ops[-2].result(0)
        out_grads = [[dout]]
        stop_gradients = [[False], [False]]
        divide_op = pir_program.global_block().ops[-1]
        with paddle.pir.core.program_guard(pir_program):
            grad_outs = call_vjp(
                divide_op,
                [[value] for value in divide_op.operands_source()],
                [[value] for value in divide_op.results()],
                out_grads,
                stop_gradients,
            )
        self.assertEqual(len(grad_outs), 2)
        self.assertEqual(
            grad_outs[0][0].get_defining_op().name(), "pd_op.divide_grad"
        )
        self.assertEqual(
            grad_outs[1][0].get_defining_op().name(), "pd_op.divide_grad"
        )
        self.assertEqual(len(pir_program.global_block().ops), 5)

    def test_sum_grad_prim(self):
        pir_program = get_ir_sum_program()
        paddle.framework.core._set_prim_backward_enabled(True)
        with paddle.pir_utils.IrGuard():
            dout = pir_program.global_block().ops[-3].result(0)
            out_grads = [[dout]]
            stop_gradients = [[False]]
            sum_op = pir_program.global_block().ops[-1]
            with paddle.pir.core.program_guard(pir_program):
                grad_outs = call_vjp(
                    sum_op,
                    [[value] for value in sum_op.operands_source()],
                    [[value] for value in sum_op.results()],
                    out_grads,
                    stop_gradients,
                )
            expand_op = pir_program.global_block().ops[-1]
            self.assertEqual(len(grad_outs), 1)
            self.assertEqual(len(pir_program.global_block().ops), 8)
            self.assertTrue(expand_op.result(0).is_same(grad_outs[0][0]))
            all_op_names = [
                "pd_op.full",
                "pd_op.full",
                "pd_op.full_int_array",
                "pd_op.sum",
                "pd_op.full_int_array",
                "pd_op.reshape",
                "pd_op.full_int_array",
                "pd_op.expand",
            ]
            for idx, op in enumerate(pir_program.global_block().ops):
                self.assertEqual(op.name(), all_op_names[idx])
            paddle.framework.core._set_prim_backward_enabled(False)

    def test_sum_grad_no_prim(self):
        pir_program = get_ir_sum_program()
        paddle.framework.core._set_prim_backward_enabled(False)
        dout = pir_program.global_block().ops[-2].result(0)
        out_grads = [[dout]]
        stop_gradients = [[False]]
        sum_op = pir_program.global_block().ops[-1]
        with paddle.pir.core.program_guard(pir_program):
            grad_outs = call_vjp(
                sum_op,
                [[value] for value in sum_op.operands_source()],
                [[value] for value in sum_op.results()],
                out_grads,
                stop_gradients,
            )
        self.assertEqual(len(grad_outs), 1)
        self.assertEqual(
            grad_outs[0][0].get_defining_op().name(), "pd_op.sum_grad"
        )
        self.assertEqual(len(pir_program.global_block().ops), 5)


if __name__ == "__main__":
    unittest.main()
