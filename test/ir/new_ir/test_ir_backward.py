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
from paddle.autograd.backward import grad

paddle.enable_static()


def get_ir_program_0():
    x = paddle.randn([4, 4])
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        x_s.stop_gradient = False
        k_s = paddle.tanh(x_s)
    newir_program = ir.translate_to_new_ir(main_program.desc)
    return newir_program


def get_ir_program_1():
    x = paddle.randn([2, 2])
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        y_s = paddle.static.data('y', [4, 4], x.dtype)
        x_s.stop_gradient = False
        z_x = paddle.tanh(y_s)
        k_s = paddle.tanh(x_s)
        out = paddle.add(z_x, k_s)
    newir_program = ir.translate_to_new_ir(main_program.desc)
    return newir_program


class TesBackward(unittest.TestCase):
    def test_1(self):
        newir_program = get_ir_program_0()
        input = newir_program.block().ops[-1].operand(0).source()
        tanh_out = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.mean(tanh_out)
            out2 = paddle.mean(tanh_out)
            input_grad = grad(out, input, out2)

        print(newir_program)
        self.assertEqual(out.get_defining_op().name(), "pd.mean")
        self.assertEqual(input_grad[0].get_defining_op().name(), "pd.tanh_grad")
        self.assertEqual(
            out.get_defining_op()
            .operands()[0]
            .source()
            .get_defining_op()
            .name(),
            "pd.tanh",
        )
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": False})

    def test_2(self):
        # test create output_grad in backward use full op
        newir_program = get_ir_program_0()
        input = newir_program.block().ops[-1].operand(0).source()
        tanh_out = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.mean(tanh_out)
            input_grad = grad(out, input)

        print(newir_program)
        self.assertEqual(newir_program.block().ops[-3].name(), "pd.full")
        self.assertEqual(input_grad[0].get_defining_op().name(), "pd.tanh_grad")
        self.assertEqual(
            input_grad[0]
            .get_defining_op()
            .operands()[1]
            .source()
            .get_defining_op()
            .name(),
            "pd.mean_grad",
        )
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": False})

    # TODO(Ruting) test add_n op when add_n api and add_grad finished
    # def test_3(self):
    #     # test add_n op
    #     newir_program = get_ir_program_1()
    #     input = newir_program.block().ops[-1].operand(0).source()
    #     tanh_out = newir_program.block().ops[-1].result(0)
    #     paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
    #     with paddle.ir.core.program_guard(newir_program):
    #         out = paddle.mean(tanh_out)
    #         input_grad = grad(out, input)

    #     print(newir_program)
    #     self.assertEqual(newir_program.block().ops[-1].name(), "pd.add_n")


if __name__ == "__main__":
    unittest.main()
