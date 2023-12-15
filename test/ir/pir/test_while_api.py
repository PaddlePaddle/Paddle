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


class TestBuildModuleWithIfOp(unittest.TestCase):
    def construct_program_with_while(self):
        main_program = paddle.static.Program()
        with paddle.pir.core.program_guard(main_program):
            i = paddle.full(
                shape=[1], fill_value=0, dtype='int64'
            )  # loop counter
            ten = paddle.full(
                shape=[1], fill_value=10, dtype='int64'
            )  # loop length
            i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])
            return main_program

    def test_while_base(self):
        main_program = self.construct_program_with_while()
        last_op = main_program.global_block().ops[-1]
        out = last_op.results()
        self.assertEqual(last_op.name(), "pd_op.while")
        self.assertEqual(len(out), 2)

    def test_while_op_vjp_interface(self):
        main_program = self.construct_program_with_while()
        while_op = main_program.global_block().ops[-1]
        self.assertEqual(while_op.name(), "pd_op.while")
        build_pipe_for_block(while_op.as_while_op().body())
        with paddle.pir.core.program_guard(main_program):
            out_grad = paddle.full(shape=[6, 1], dtype='float32', fill_value=3)
            # check vjp interface for while_op
            while_input = [
                [input] for input in get_used_external_value(while_op)
            ]
            self.assertEqual(len(while_input), 4)
            while_input_stop_graditents = [[True], [False], [True], [True]]
            while_output = [while_op.results()]
            while_output_grad = [[out_grad, out_grad]]
            self.assertEqual(has_vjp(while_op), True)
            grad_outs = call_vjp(
                while_op,
                while_input,
                while_output,
                while_output_grad,
                while_input_stop_graditents,
            )

            self.assertEqual(grad_outs[0][0], None)

            while_grad_op = grad_outs[1][0].get_defining_op()
            self.assertEqual(while_grad_op.name(), "pd_op.while")
            while_grad_output = while_grad_op.results()
            self.assertEqual(len(while_grad_output), 1)


if __name__ == "__main__":
    unittest.main()
