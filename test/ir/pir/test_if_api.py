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


def true_func():
    a = paddle.full(shape=[1, 2], dtype='float32', fill_value=1)
    b = paddle.full(shape=[2, 3], dtype='int64', fill_value=1)
    return a, b


def false_func():
    a = paddle.full(shape=[1, 2], dtype='float32', fill_value=3)
    b = paddle.full(shape=[2, 3], dtype='int64', fill_value=2)
    return a, b


class TestBuildModuleWithIfOp(unittest.TestCase):
    def construct_program_with_if(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
            y = paddle.static.data(name="y", shape=[6, 1], dtype="float32")
            x.stop_gradient = False
            y.stop_gradient = False
            paddle.static.nn.cond(x < y, lambda: x + y, lambda: x - y)
        return main_program

    def test_if_with_single_output(self):
        main_program = self.construct_program_with_if()
        if_op = main_program.global_block().ops[-1]
        self.assertEqual(if_op.name(), "pd_op.if")
        self.assertEqual(len(if_op.results()), 1)
        value_list = get_used_external_value(if_op)
        self.assertEqual(len(value_list), 3)
        self.assertTrue(value_list[0].is_same(if_op.operand_source(0)))

    def test_if_with_multiple_output(self):
        main_program = self.construct_program_with_if()
        cond_value = main_program.global_block().ops[-1].operand_source(0)
        with paddle.pir.core.program_guard(main_program):
            paddle.static.nn.cond(cond_value, true_func, false_func)
        last_op = main_program.global_block().ops[-1]
        out = last_op.results()
        self.assertEqual(last_op.name(), "pd_op.if")
        self.assertEqual(len(out), 2)

        # check Operation::as_if_op interface
        if_op = last_op.as_if_op()
        true_block = if_op.true_block()
        self.assertEqual(len(true_block), 3)

        # check build_pipe_for_block interface
        build_pipe_for_block(true_block)
        self.assertEqual(len(true_block), 4)

        # check Operation::blocks interface
        block_list = []
        for block in out[0].get_defining_op().blocks():
            block_list.append(block)
        self.assertEqual(len(block_list), 2)
        self.assertEqual(block_list[0], true_block)
        self.assertEqual(block_list[1], if_op.false_block())

    def test_if_op_vjp_interface(self):
        main_program = self.construct_program_with_if()
        if_op = main_program.global_block().ops[-1]
        self.assertEqual(if_op.name(), "pd_op.if")
        build_pipe_for_block(if_op.as_if_op().true_block())
        with paddle.pir.core.program_guard(main_program):
            out_grad = paddle.full(shape=[6, 1], dtype='float32', fill_value=3)
            # check vjp interface for if_op
            if_input = [[input] for input in get_used_external_value(if_op)]
            if_input_stop_gradients = [[False], [False], [True]]
            if_output = [if_op.results()]
            if_output_grad = [[out_grad]]
            self.assertEqual(has_vjp(if_op), True)
            grad_outs = call_vjp(
                if_op,
                if_input,
                if_output,
                if_output_grad,
                if_input_stop_gradients,
            )

            self.assertEqual(len(grad_outs), len(if_input) - 1)

            if_grad_op = grad_outs[1][0].get_defining_op()
            self.assertEqual(if_grad_op.name(), "pd_op.if")
            with if_grad_op.as_if_op().true_block():
                # check vjp interface for tupe_push_op
                push_op = if_op.as_if_op().true_block().ops[-2]
                self.assertEqual(push_op.name(), "cf.tuple_push")
                self.assertEqual(has_vjp(push_op), True)
                print([[value] for value in push_op.operands_source()])
                pop_outs = call_vjp(
                    push_op,
                    [[value] for value in push_op.operands_source()],
                    [[]],
                    [[]],
                    [[True], [False]],
                )
                self.assertEqual(len(pop_outs), 2)
                self.assertEqual(
                    pop_outs[1][0].get_defining_op().name(), "cf.tuple_pop"
                )

    def test_if_op_backward(self):
        main_program = self.construct_program_with_if()
        dataop0 = main_program.global_block().ops[0]
        dataop1 = main_program.global_block().ops[1]
        if_op = main_program.global_block().ops[-1]
        self.assertEqual(if_op.name(), "pd_op.if")
        with paddle.pir.core.program_guard(main_program):
            self.assertEqual(
                main_program.global_block().ops[-2].result(0).stop_gradient,
                True,
            )
            self.assertEqual(if_op.result(0).stop_gradient, False)
            # check vjp interface for if_op
            grad_outs = grad(
                if_op.results(),
                [dataop0.result(0), dataop1.result(0)],
            )

            self.assertEqual(grad_outs[0].get_defining_op().name(), "pd_op.if")
            self.assertEqual(grad_outs[1].get_defining_op().name(), "pd_op.if")


if __name__ == "__main__":
    unittest.main()
