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

import numpy as np

import paddle
from paddle import pir
from paddle.autograd.backward_utils import ValueDict, ValueSet
from paddle.autograd.ir_backward import grad

paddle.enable_static()


def get_ir_program_0():
    paddle.enable_static()
    with paddle.pir_utils.OldIrGuard():
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
                pir_program.global_block().ops[-3].name(), "pd_op.mean"
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
    with paddle.pir_utils.OldIrGuard():
        x = paddle.randn([2, 2])
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            x_s = paddle.static.data('x', [4, 4], x.dtype)
            x_s.stop_gradient = False

            k_s = paddle.tanh(x_s)
            z_x = paddle.tanh(x_s)
            out = paddle.add(z_x, k_s)
        pir_program = pir.translate_to_pir(main_program.desc)
        return pir_program


class TesBackward_2(unittest.TestCase):
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
    with paddle.pir_utils.OldIrGuard():
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


class TestBackward_4(unittest.TestCase):
    def test_basic_network(self):
        if not paddle.framework.in_pir_mode():
            return
        program = paddle.static.default_main_program()
        with paddle.static.program_guard(program):
            x = paddle.randn((2, 2))
            x.stop_gradient = False
            b = paddle.to_tensor([12])
            grad_x = 0
            double_x = x * 2
            pred = b > 0

            def true_func():
                y = double_x * 3
                out = grad(y, x)
                filted_dx = [dxi for dxi in out if dxi is not None]
                grad_x = filted_dx
                return grad_x

            def false_func():
                y = double_x * 4
                out = grad(y, x)
                filted_dx = [dxi for dxi in out if dxi is not None]
                grad_x = filted_dx
                return grad_x

            out = paddle.static.nn.cond(pred, true_func, false_func)

            place = (
                paddle.base.CUDAPlace(0)
                if paddle.base.core.is_compiled_with_cuda()
                else paddle.base.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            (grad_x,) = exe.run(program, fetch_list=[out])
            res = np.full([2, 2], 6.0, dtype='float32')
            self.assertEqual((grad_x == res).all(), True)


class TestBackward_5(unittest.TestCase):
    def test_skip_vjp(self):
        if not paddle.framework.in_pir_mode():
            return
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            x = paddle.static.data('x', [4, 4], 'float32')
            x.stop_gradient = True
            y = paddle.nn.functional.relu(x)
            y.stop_gradient = False
            z = paddle.nn.functional.relu(y)
            loss = paddle.mean(z)

            paddle.autograd.ir_backward.append_backward(loss)
        relu_grad_number = 0
        for op in program.global_block().ops:
            if op.name() == "pd_op.relu_grad":
                relu_grad_number += 1

        self.assertEqual(relu_grad_number, 1)


class TestValueSet(unittest.TestCase):
    def setUp(self) -> None:
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                self.x = paddle.static.data('x', [2, 3])
                self.y = paddle.static.data('y', [4, 5])

    def test_copy(self):
        a = ValueSet([self.x, self.y])
        b = a.copy()
        self.assertNotEqual(id(a), id(b))
        self.assertTrue(len(a) == len(b))

    def test_or(self):
        a = ValueSet([self.x])
        b = ValueSet([self.y])
        c = a | b
        self.assertTrue(len(c) == 2)


class TestValueDict(unittest.TestCase):
    def setUp(self) -> None:
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                self.x = paddle.static.data('x', [2, 3])
                self.y = paddle.static.data('y', [4, 5])

    def test_init(self):
        a = ValueDict()
        a[self.x] = 'x'
        a[self.y] = 'y'
        b = ValueDict(a)
        self.assertTrue(len(b) == 2)

    def test_bool(self):
        a = ValueDict()
        self.assertFalse(bool(a))

    def test_getitem_and_pop_error(self):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [2, 3])
                y = paddle.static.data('y', [4, 5])
                a = ValueDict()
                a[x] = 'x'
                self.assertRaises(KeyError, a.__getitem__, y)
                self.assertRaises(KeyError, a.pop, y)

    def test_update(self):
        a = ValueDict()
        a[self.x] = 'x'
        b = ValueDict()
        b[self.y] = 'y'
        a.update(b)
        self.assertTrue(a[self.y] == 'y')


if __name__ == "__main__":
    unittest.main()
