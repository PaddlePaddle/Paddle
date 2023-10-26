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


def get_ir_program():
    paddle.enable_static()
    x = paddle.randn([4, 4])
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        x_s.stop_gradient = False
        y_s = paddle.matmul(x_s, x_s)
        y_s = paddle.add(x_s, y_s)
        y_s = paddle.tanh(y_s)
    newir_program = pir.translate_to_new_ir(main_program.desc)
    return newir_program


class TestBuildOp(unittest.TestCase):
    def test_build_mean_op(self):
        newir_program = get_ir_program()
        tanh_out = newir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            out = paddle.mean(tanh_out)
            self.assertEqual(out.get_defining_op().name(), "pd_op.mean")
            self.assertEqual(
                out.get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.tanh",
            )


class TestBuildOp2(unittest.TestCase):
    def test_build_add_n_op(self):
        newir_program = get_ir_program()
        tanh_out = newir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            out1 = paddle.mean(tanh_out)
            out2 = paddle.mean(tanh_out)
            out = paddle.add_n([out1, out2])
            self.assertEqual(out.get_defining_op().name(), "pd_op.add_n")
            self.assertEqual(
                out.get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "builtin.combine",
            )


class TestBuildOp3(unittest.TestCase):
    def test_insertion_point(self):
        newir_program = get_ir_program()
        with paddle.pir_utils.IrGuard():
            add_op = newir_program.global_block().ops[-2]
            tanh_op = newir_program.global_block().ops[-1]
            add_out = add_op.result(0)
            tanh_operand = tanh_op.operands()[0]

            with paddle.pir.core.program_guard(newir_program):
                pir.set_insertion_point(tanh_op)
                full_out = paddle.tensor.fill_constant(
                    shape=[4, 4], dtype="float", value=2
                )
                divide_out = paddle.divide(full_out, full_out)
                sum_out = paddle.sum(divide_out)
                out = paddle.mean(sum_out)
                tanh_operand.set_source(out)

            print(newir_program)
            self.assertEqual(
                tanh_operand.source().get_defining_op().name(), "pd_op.mean"
            )


class TestBuildOp4(unittest.TestCase):
    def test_build_concat_op(self):
        newir_program = get_ir_program()
        tanh_out = newir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            out = paddle.concat([tanh_out, tanh_out], 0)
            self.assertEqual(out.get_defining_op().name(), "pd_op.concat")
            self.assertEqual(
                out.get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "builtin.combine",
            )


class TestBuildOp5(unittest.TestCase):
    def test_build_split_op(self):
        newir_program = get_ir_program()
        tanh_out = newir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            out = paddle.split(tanh_out, [2, 2], 0)
            self.assertEqual(out[0].get_defining_op().name(), "builtin.split")
            self.assertEqual(
                out[0]
                .get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.split",
            )


if __name__ == "__main__":
    unittest.main()
