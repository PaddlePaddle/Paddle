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
        self.assertEqual(out.stop_gradient, False)
        self.assertEqual(last_op.name(), "pd_op.while")
        self.assertEqual(len(out), 2)

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


if __name__ == "__main__":
    unittest.main()
