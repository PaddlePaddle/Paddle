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

paddle.enable_static()


def get_ir_program():
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
    newir_program = ir.translate_to_new_ir(main_program.desc)
    return newir_program


class TestBuildOp(unittest.TestCase):
    def test_build_op(self):
        newir_program = get_ir_program()
        tanh_out = newir_program.block().get_ops()[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.mean(tanh_out)
        print(newir_program)
        self.assertEqual(out.get_defining_op().name(), "pd.mean")
        self.assertEqual(
            out.get_defining_op()
            .operands()[0]
            .source()
            .get_defining_op()
            .name(),
            "pd.tanh",
        )

    def test_insertion_point(self):
        newir_program = get_ir_program()
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        add_op = newir_program.block().get_ops()[-2]
        tanh_op = newir_program.block().get_ops()[-1]
        add_out = add_op.result(0)
        tanh_operand = tanh_op.operands()[0]

        with paddle.ir.core.program_guard(newir_program):
            ir.set_insertion_point(tanh_op)
            out = paddle.mean(add_out)
            tanh_operand.set_source(out)

        print(newir_program)
        self.assertEqual(
            tanh_operand.source().get_defining_op().name(), "pd.mean"
        )


if __name__ == "__main__":
    unittest.main()
