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


class TesBackward_1(unittest.TestCase):
    def test_grad(self):
        newir_program = get_ir_program_0()
        input = newir_program.block().ops[-1].operand(0).source()
        tanh_out = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.mean(tanh_out)
            out2 = paddle.mean(tanh_out)
            input_grad = grad(out, input, out2)

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

    def test_full(self):
        # test create output_grad in backward use full op
        newir_program = get_ir_program_0()
        input = newir_program.block().ops[-1].operand(0).source()
        tanh_out = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.mean(tanh_out)
            input_grad = grad(out, input)

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

    def test_no_grad_set(self):
        # test create output_grad in backward use full op
        newir_program = get_ir_program_0()
        input = newir_program.block().ops[-1].operand(0).source()
        tanh_out = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.mean(tanh_out)
            input_grad = grad(out, input, no_grad_vars=[input])

        self.assertEqual(newir_program.block().ops[-1].name(), "pd.mean")
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": False})

    def test_split(self):
        # test create output_grad in backward use full op
        newir_program = get_ir_program_0()
        input = newir_program.block().ops[-1].operand(0).source()
        tanh_out = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.split(tanh_out, [2, 2], 0)
            input_grad = grad(out, input)

        ops_name = [
            "pd.data",
            "pd.tanh",
            "pd.full_int_array",
            "pd.full",
            "pd.split",
            "builtin.split",
            "pd.full",
            "builtin.combine",
            "pd.split_grad",
            "pd.tanh_grad",
        ]
        for i, op in enumerate(newir_program.block().ops):
            self.assertEqual(op.name(), ops_name[i])
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": False})


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
        y_s.stop_gradient = False

        k_s = paddle.tanh(x_s)
        z_x = paddle.tanh(x_s)
        out = paddle.add(z_x, k_s)
    newir_program = ir.translate_to_new_ir(main_program.desc)
    return newir_program


class TesBackward_2(unittest.TestCase):
    def test_add_n(self):
        newir_program = get_ir_program_1()
        input_x = newir_program.block().ops[-3].operand(0).source()

        add_out = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.mean(add_out)
            input_grad = grad(out, input_x)

        self.assertEqual(newir_program.block().ops[-1].name(), "pd.add_n")
        self.assertEqual(
            newir_program.block().ops[-2].name(), "builtin.combine"
        )
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": False})

    def test_concat(self):
        newir_program = get_ir_program_1()
        input_x = newir_program.block().ops[-3].operand(0).source()

        add_out = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            out = paddle.concat([add_out, add_out])
            input_grad = grad(out, input_x)

        ops_name = [
            "pd.data",
            "pd.data",
            "pd.tanh",
            "pd.tanh",
            "pd.add",
            "builtin.combine",
            "pd.full",
            "pd.concat",
            "pd.full",
            "builtin.combine",
            "pd.concat_grad",
            "builtin.split",
            "builtin.combine",
            "pd.add_n",
            "pd.add_grad",
            "pd.tanh_grad",
            "pd.tanh_grad",
            "builtin.combine",
            "pd.add_n",
        ]
        for i, op in enumerate(newir_program.block().ops):
            self.assertEqual(op.name(), ops_name[i])

        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": False})


def get_ir_program_2():
    x = paddle.randn([2, 2])
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        x_s.stop_gradient = False
        k_s = paddle.sum(x_s, axis=(-1,), keepdim=False)
    newir_program = ir.translate_to_new_ir(main_program.desc)
    return newir_program


class TestBackward_3(unittest.TestCase):
    def test_basic_network(self):
        newir_program = get_ir_program_2()
        x = newir_program.block().ops[-1].operand(0).source()
        sum_x = newir_program.block().ops[-1].result(0)
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        with paddle.ir.core.program_guard(newir_program):
            norm = paddle.tensor.fill_constant(
                shape=[],
                value=1.0,
                dtype=sum_x.dtype,
            )
            res = paddle.divide(sum_x, norm)
            input_grad = grad(res, x)

        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": False})


if __name__ == "__main__":
    unittest.main()
