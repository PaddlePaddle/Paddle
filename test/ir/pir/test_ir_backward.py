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
from paddle.autograd.ir_backward import grad

paddle.enable_static()


def get_ir_program_0():
    paddle.enable_static()
    x = paddle.randn([4, 4])
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        x_s.stop_gradient = False
        k_s = paddle.tanh(x_s)
    pir_program = pir.translate_to_pir(main_program.desc)
    return pir_program


class TesBackward_1(unittest.TestCase):
    def tearDown(self) -> None:
        paddle.framework.set_flags({"FLAGS_enable_pir_api": False})

    def test_grad(self):
        pir_program = get_ir_program_0()
        input = pir_program.global_block().ops[-1].operand(0).source()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.mean(tanh_out)
            out2 = paddle.mean(tanh_out)
            input_grad = grad(out, input, out2)

            self.assertEqual(out.get_defining_op().name(), "pd_op.mean")
            self.assertEqual(
                input_grad[0].get_defining_op().name(), "pd_op.tanh_grad"
            )
            self.assertEqual(
                out.get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.tanh",
            )

    def test_full(self):
        # test create output_grad in backward use full op
        pir_program = get_ir_program_0()
        input = pir_program.global_block().ops[-1].operand(0).source()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.mean(tanh_out)
            input_grad = grad(out, input)
            self.assertEqual(
                pir_program.global_block().ops[-3].name(), "pd_op.full_like"
            )
            self.assertEqual(
                input_grad[0].get_defining_op().name(), "pd_op.tanh_grad"
            )
            self.assertEqual(
                input_grad[0]
                .get_defining_op()
                .operands()[1]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.mean_grad",
            )

    def test_no_grad_set(self):
        # test create output_grad in backward use full op
        pir_program = get_ir_program_0()
        input = pir_program.global_block().ops[-1].operand(0).source()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.mean(tanh_out)
            input_grad = grad(out, input, no_grad_vars=[input])
            self.assertEqual(
                pir_program.global_block().ops[-1].name(), "pd_op.full"
            )

    def test_split(self):
        # test create output_grad in backward use full op
        pir_program = get_ir_program_0()
        input = pir_program.global_block().ops[-1].operand(0).source()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.split(tanh_out, [2, 2], 0)
            input_grad = grad(out, input)
            ops_name = [
                "pd_op.data",
                "pd_op.tanh",
                "pd_op.full_int_array",
                "pd_op.full",
                "pd_op.split",
                "builtin.split",
                "pd_op.full",
                "pd_op.full_like",
                "pd_op.full",
                "pd_op.full_like",
                "builtin.combine",
                "pd_op.concat",
                "pd_op.tanh_grad",
            ]
            for i, op in enumerate(pir_program.global_block().ops):
                self.assertEqual(op.name(), ops_name[i])


def get_ir_program_1():
    paddle.enable_static()
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
    pir_program = pir.translate_to_pir(main_program.desc)
    return pir_program


class TesBackward_2(unittest.TestCase):
    def tearDown(self) -> None:
        paddle.framework.set_flags({"FLAGS_enable_pir_api": False})

    def test_add_n(self):
        pir_program = get_ir_program_1()
        input_x = pir_program.global_block().ops[-3].operand(0).source()

        add_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.mean(add_out)
            input_grad = grad(out, input_x)

            self.assertEqual(
                pir_program.global_block().ops[-1].name(), "pd_op.add_n"
            )
            self.assertEqual(
                pir_program.global_block().ops[-1].name(), "pd_op.add_n"
            )
            self.assertEqual(
                pir_program.global_block().ops[-2].name(), "builtin.combine"
            )

    def test_concat(self):
        pir_program = get_ir_program_1()
        input_x = pir_program.global_block().ops[-3].operand(0).source()

        add_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.concat([add_out, add_out])
            input_grad = grad(out, input_x)
        ops_name = [
            "pd_op.data",
            "pd_op.data",
            "pd_op.tanh",
            "pd_op.tanh",
            "pd_op.add",
            "pd_op.full",
            "builtin.combine",
            "pd_op.concat",
            "pd_op.full",
            "pd_op.full_like",
            "builtin.combine",
            "pd_op.concat_grad",
            "builtin.split",
            "builtin.combine",
            "pd_op.add_n",
            "pd_op.add_grad",
            "pd_op.tanh_grad",
            "pd_op.tanh_grad",
            "builtin.combine",
            "pd_op.add_n",
        ]
        for i, op in enumerate(pir_program.global_block().ops):
            self.assertEqual(op.name(), ops_name[i])


def get_ir_program_2():
    paddle.enable_static()
    x = paddle.randn([2, 2])
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        x_s.stop_gradient = False
        k_s = paddle.sum(x_s, axis=(-1,), keepdim=False)
    pir_program = pir.translate_to_pir(main_program.desc)
    return pir_program


class TestBackward_3(unittest.TestCase):
    def tearDown(self) -> None:
        paddle.framework.set_flags({"FLAGS_enable_pir_api": False})

    def test_basic_network(self):
        pir_program = get_ir_program_2()
        x = pir_program.global_block().ops[-1].operand(0).source()
        sum_x = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            norm = paddle.tensor.fill_constant(
                shape=[],
                value=1.0,
                dtype=sum_x.dtype,
            )
            res = paddle.divide(sum_x, norm)
            input_grad = grad(res, x)


class TestBackward_refrash_stopgradients(unittest.TestCase):
    def test_refreash_stopgradients(self):
        import numpy as np

        program = paddle.pir.core.default_main_program()
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(program):
            data1 = paddle.static.data('data1', [3, 4, 5], np.float32)
            data2 = paddle.static.data('data2', [3, 4, 5], np.float32)
            out = paddle.add_n([data1, data2])
            data1_arr = np.random.uniform(-1, 1, data1.shape).astype(np.float32)
            data2_arr = np.random.uniform(-1, 1, data2.shape).astype(np.float32)
            self.assertEqual(
                program.global_block().ops[3].result(0).stop_gradient, True
            )

            data1.stop_gradient = False
            data2.stop_gradient = False

            dout = grad(out, [data1, data2])
            self.assertEqual(
                program.global_block().ops[3].result(0).stop_gradient, False
            )


if __name__ == "__main__":
    unittest.main()
