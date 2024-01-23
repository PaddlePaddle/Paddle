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
from paddle.autograd.ir_backward import grad
from paddle.base.core import call_vjp, has_vjp
from paddle.base.libpaddle.pir import (
    build_pipe_for_block,
    get_used_external_value,
)

paddle.enable_static()


def cond(i, ten):
    return i < ten


def body(i, ten):
    i = i + 1
    (i,) = paddle.static.nn.while_loop(
        lambda p: p < ten, lambda p: [p + 3], [i]
    )
    return [i, ten]


class TestBuildModuleWithWhileOp(unittest.TestCase):
    def construct_program_with_while(self):
        main_program = paddle.static.Program()
        with paddle.pir.core.program_guard(main_program):
            i = paddle.full(
                shape=[1], fill_value=0, dtype='int64'
            )  # loop counter
            ten = paddle.full(
                shape=[1], fill_value=10, dtype='int64'
            )  # loop length
            i.stop_gradient = False
            i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])
            return main_program

    def test_while_base(self):
        main_program = self.construct_program_with_while()
        last_op = main_program.global_block().ops[-1]
        out = last_op.results()
        self.assertEqual(out[0].stop_gradient, False)
        self.assertEqual(last_op.name(), "pd_op.while")
        self.assertEqual(len(out), 1)

    def test_get_used_external_value(self):
        main_program = paddle.static.Program()
        with paddle.pir.core.program_guard(main_program):
            i = paddle.full(shape=[1], fill_value=0)
            x = paddle.full(shape=[1], fill_value=10)
            y = paddle.full(shape=[1], fill_value=5)
            # i, x = paddle.static.nn.while_loop(cond, body, [i, ten])
            paddle.static.nn.while_loop(
                lambda p, q: p < q, lambda p, q: [p + y, q + i], [i, x]
            )
        while_op = main_program.global_block().ops[-1]
        self.assertEqual(while_op.name(), "pd_op.while")
        body_block = while_op.as_while_op().body()
        operand_source = while_op.operands_source()
        # 【cond, i , x】
        self.assertEqual(len(operand_source), 3)
        self.assertTrue(operand_source[1].is_same(i))
        self.assertTrue(operand_source[2].is_same(x))

        block_external_values = get_used_external_value(body_block)
        # 【y, i】
        self.assertEqual(len(block_external_values), 2)
        self.assertTrue(block_external_values[0].is_same(y))
        self.assertTrue(block_external_values[1].is_same(i))

        op_external_values = get_used_external_value(while_op)
        # 【cond, i , x， y】
        self.assertEqual(len(op_external_values), 4)
        self.assertTrue(op_external_values[1].is_same(i))
        self.assertTrue(op_external_values[2].is_same(x))
        self.assertTrue(op_external_values[3].is_same(y))

    def test_while_op_vjp_interface(self):
        main_program = self.construct_program_with_while()
        while_op = main_program.global_block().ops[-1]
        self.assertEqual(while_op.name(), "pd_op.while")
        body_block = while_op.as_while_op().body()
        build_pipe_for_block(body_block)
        with paddle.pir.core.program_guard(main_program):
            out_grad = paddle.full(shape=[6, 1], dtype='float32', fill_value=3)
            # check vjp interface for while_op
            while_input = [[input] for input in while_op.operands_source()] + [
                [input] for input in get_used_external_value(body_block)
            ]
            self.assertEqual(len(while_input), 4)
            while_input_stop_gradients = [[True], [False], [True], [True]]
            while_output = [[value] for value in while_op.results()]
            while_output_grad = [[out_grad], [out_grad], [out_grad]]
            self.assertEqual(has_vjp(while_op), True)
            grad_outs = call_vjp(
                while_op,
                while_input,
                while_output,
                while_output_grad,
                while_input_stop_gradients,
            )

            self.assertEqual(grad_outs[0][0], None)

            while_grad_op = grad_outs[1][0].get_defining_op()
            self.assertEqual(while_grad_op.name(), "pd_op.while")
            while_grad_output = while_grad_op.results()
            self.assertEqual(len(while_grad_output), 1)

    def test_while_base_backward(self):
        main_program = self.construct_program_with_while()
        full_op1 = main_program.global_block().ops[0]
        while_op = main_program.global_block().ops[-1]
        with paddle.pir.core.program_guard(main_program):
            out = while_op.result(0) + 1
            grad_outs = grad(
                out,
                [full_op1.result(0)],
            )

            self.assertEqual(
                grad_outs[0].get_defining_op().name(), "pd_op.while"
            )


def cond2(i, j, ten):
    return i < ten


def body2(i, j, ten):
    i = i + j
    return [i, j, ten]


class TestBuildModuleWithWhile2Op(unittest.TestCase):
    def test_backward(self):
        main_program = paddle.static.Program()
        with paddle.pir.core.program_guard(main_program):
            i = paddle.full(
                shape=[1], fill_value=0, dtype='int64'
            )  # loop counter
            j = paddle.full(
                shape=[1], fill_value=2, dtype='int64'
            )  # loop counter
            ten = paddle.full(
                shape=[1], fill_value=10, dtype='int64'
            )  # loop length
            i.stop_gradient = False
            j.stop_gradient = False
            i_, j_, ten_ = paddle.static.nn.while_loop(
                cond2, body2, [i, j, ten]
            )
            out = i_ - j_

            grad_outs = grad(
                out,
                [i, j],
            )
            self.assertEqual(
                grad_outs[0].get_defining_op().name(), "pd_op.while"
            )
            self.assertEqual(
                main_program.global_block()
                .ops[-1]
                .as_while_op()
                .body()
                .ops[-4]
                .name(),
                "cf.has_elements",
            )

            self.assertEqual(
                main_program.global_block()
                .ops[-1]
                .as_while_op()
                .body()
                .ops[-5]
                .name(),
                "pd_op.add_grad",
            )

    def test_backward_with_loop_var_same_to_extra_var(self):
        main_program = paddle.static.Program()
        with paddle.pir.core.program_guard(main_program):
            i = paddle.full(shape=[1], fill_value=0)
            x = paddle.full(shape=[1], fill_value=5)
            y = paddle.full(shape=[1], fill_value=10)
            i.stop_gradient = False
            x.stop_gradient = False
            y.stop_gradient = False
            new_i, new_x = paddle.static.nn.while_loop(
                lambda p, q: p < q, lambda p, q: [p + y, q + x], [i, x]
            )

            out = new_i - new_x
            grad_outs = grad(out, [i, x, y])

            self.assertEqual(
                grad_outs[0].get_defining_op().name(), "pd_op.while"
            )
            self.assertEqual(
                grad_outs[1].get_defining_op().name(), "pd_op.add_n"
            )
            self.assertEqual(
                grad_outs[2].get_defining_op().name(), "pd_op.while"
            )
            self.assertEqual(
                main_program.global_block()
                .ops[-3]
                .as_while_op()
                .body()
                .ops[-1]
                .operand_source(1)
                .get_defining_op()
                .name(),
                "pd_op.add_grad",
            )


if __name__ == "__main__":
    unittest.main()
