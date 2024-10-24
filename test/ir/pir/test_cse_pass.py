# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from contextlib import contextmanager

import paddle

paddle.enable_static()


@contextmanager
def program_scope_guard():
    place = paddle.framework.core.Place()
    place.set_place(paddle.CPUPlace())
    new_scope = paddle.static.Scope()
    main_program = paddle.static.Program()
    with paddle.static.scope_guard(new_scope):
        with paddle.static.program_guard(main_program):
            yield main_program


@contextmanager
def flag_guard(flag_name, flag_value):
    old_value = paddle.get_flags(flag_name)[flag_name]
    paddle.set_flags({flag_name: flag_value})
    try:
        yield
    finally:
        paddle.set_flags({flag_name: old_value})


def walk_block(block, fn):
    for op in block.ops:
        fn(op)
        for subblock in op.blocks():
            walk_block(subblock, fn)


def count_op(program, op_name):
    count = 0

    def count_fn(op):
        nonlocal count
        if op.name() == op_name:
            count += 1

    walk_block(program.global_block(), count_fn)
    return count


class AssertOpCountEqualMixin:
    def assert_op_count_equal(self, program, op_count_map):
        for op_name, expected_count in op_count_map.items():
            actual_count = count_op(program, op_name)
            self.assertEqual(
                actual_count,
                expected_count,
                msg=f"Expect program has {expected_count} {op_name}, but got {actual_count} {op_name}",
            )


class TestCSEBasic(unittest.TestCase, AssertOpCountEqualMixin):
    def test_basic(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")

            # Expr1
            a = x1 + x2

            # Expr2
            b = x1 + x2

            self.assert_op_count_equal(main_program, {"pd_op.add": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.add": 1})

    def test_basic2(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")
            x3 = paddle.static.data("x3", [2, 2], dtype="float32")

            def expr(x1, x2, x3):
                a = x1 + x2
                b = a + x3
                return b

            out1 = expr(x1, x2, x3)
            out2 = expr(x1, x2, x3)

            self.assert_op_count_equal(main_program, {"pd_op.add": 4})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.add": 2})

    def test_replace_full_with_same_attr(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.full([2, 2], 0, dtype="float32")
            x2 = paddle.full([2, 2], 0, dtype="float32")

            self.assert_op_count_equal(main_program, {"pd_op.full": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.full": 1})

    def test_skip_replace_full_with_different_value(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.full([2, 2], 0, dtype="float32")
            x2 = paddle.full([2, 2], 1, dtype="float32")

            self.assert_op_count_equal(main_program, {"pd_op.full": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.full": 2})

    def test_skip_replace_full_with_different_dtype(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.full([2, 2], 0, dtype="float32")
            x2 = paddle.full([2, 2], 0, dtype="int32")

            self.assert_op_count_equal(main_program, {"pd_op.full": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.full": 2})

    def test_skip_replace_full_with_different_shape(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.full([2, 2], 0, dtype="float32")
            x2 = paddle.full([3, 3], 0, dtype="float32")

            self.assert_op_count_equal(main_program, {"pd_op.full": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.full": 2})

    def test_complex_computation(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")
            x3 = paddle.static.data("x3", [2, 2], dtype="float32")

            def expr(x1, x2, x3):
                a = x1 + x2
                b = a * x3
                c = b**a
                d = c / x3
                e = d - x1
                return e

            repeat_n = 10
            for _ in range(repeat_n):
                out = expr(x1, x2, x3)

            self.assert_op_count_equal(
                main_program,
                {
                    "pd_op.add": repeat_n,
                    "pd_op.multiply": repeat_n,
                    "pd_op.elementwise_pow": repeat_n,
                    "pd_op.divide": repeat_n,
                    "pd_op.subtract": repeat_n,
                },
            )
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(
                main_program,
                {
                    "pd_op.add": 1,
                    "pd_op.multiply": 1,
                    "pd_op.elementwise_pow": 1,
                    "pd_op.divide": 1,
                    "pd_op.subtract": 1,
                },
            )


class TestCSECommutative(unittest.TestCase, AssertOpCountEqualMixin):
    def test_commutative(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")

            # Expr1
            a = x1 + x2

            # Expr2
            b = x2 + x1

            self.assert_op_count_equal(main_program, {"pd_op.add": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.add": 1})

    def test_complex_computation_commutative(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")
            x3 = paddle.static.data("x3", [2, 2], dtype="float32")

            def expr(x1, x2, x3):
                a = x1 + x2
                b = x2 * x1
                c = paddle.maximum(b, a)
                d = paddle.minimum(c, x3)
                x3 = paddle.cast(x3, 'bool')
                e = paddle.logical_and(d, c)
                f = paddle.logical_or(e, x3)
                g = paddle.logical_xor(f, e)
                h = paddle.equal(g, x3)
                i = paddle.not_equal(h, x3)
                j = paddle.bitwise_or(i, h)
                k = paddle.bitwise_xor(j, i)
                l = paddle.bitwise_and(k, j)
                return l

            repeat_n = 10
            expr_per_repeat = 2
            for _ in range(repeat_n):
                out1 = expr(x1, x2, x3)
                out2 = expr(x2, x1, x3)

            self.assert_op_count_equal(
                main_program,
                {
                    "pd_op.add": repeat_n * expr_per_repeat,
                    "pd_op.multiply": repeat_n * expr_per_repeat,
                    "pd_op.maximum": repeat_n * expr_per_repeat,
                    "pd_op.minimum": repeat_n * expr_per_repeat,
                    "pd_op.logical_and": repeat_n * expr_per_repeat,
                    "pd_op.logical_or": repeat_n * expr_per_repeat,
                    "pd_op.logical_xor": repeat_n * expr_per_repeat,
                    "pd_op.equal": repeat_n * expr_per_repeat,
                    "pd_op.not_equal": repeat_n * expr_per_repeat,
                    "pd_op.bitwise_or": repeat_n * expr_per_repeat,
                    "pd_op.bitwise_xor": repeat_n * expr_per_repeat,
                    "pd_op.bitwise_and": repeat_n * expr_per_repeat,
                },
            )
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(
                main_program,
                {
                    "pd_op.add": 1,
                    "pd_op.multiply": 1,
                    "pd_op.maximum": 1,
                    "pd_op.minimum": 1,
                    "pd_op.logical_and": 1,
                    "pd_op.logical_or": 1,
                    "pd_op.logical_xor": 1,
                    "pd_op.equal": 1,
                    "pd_op.not_equal": 1,
                    "pd_op.bitwise_or": 1,
                    "pd_op.bitwise_xor": 1,
                    "pd_op.bitwise_and": 1,
                },
            )


class TestCSESubBlock(unittest.TestCase, AssertOpCountEqualMixin):
    def test_subblock(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")
            cond = paddle.static.data("cond", [], dtype="bool")
            loop_var = paddle.static.data("i", [], dtype="int32")

            def expr(x1, x2):
                return x1 * x2

            # Expr1 in global block
            a = expr(x1, x2)

            # Expr2 in subblock
            def loop_body(loop_var):
                tmp_var = expr(x1, x2)
                return [loop_var + 1]

            def get_cond(loop_var):
                return cond

            paddle.static.nn.while_loop(get_cond, loop_body, [loop_var])

            self.assert_op_count_equal(main_program, {"pd_op.multiply": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.multiply": 1})

    def test_nested_subblock(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")
            cond = paddle.static.data("cond", [], dtype="bool")
            loop_var = paddle.static.data("i", [], dtype="int32")

            def expr(x1, x2):
                return x1 * x2

            # Expr1 in global block
            a = expr(x1, x2)

            # Expr2 in outer subblock
            def outer_loop_body(loop_var):
                # Expr3 in inner subblock
                def inner_loop_body(loop_var):
                    inner_tmp_var = expr(x1, x2)
                    return [loop_var + 1]

                def inner_get_cond(loop_var):
                    return cond

                outer_tmp_var = expr(x1, x2)
                paddle.static.nn.while_loop(
                    inner_get_cond, inner_loop_body, [loop_var]
                )
                return [loop_var + 1]

            def outer_get_cond(loop_var):
                return cond

            paddle.static.nn.while_loop(
                outer_get_cond, outer_loop_body, [loop_var]
            )

            self.assert_op_count_equal(main_program, {"pd_op.multiply": 3})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.multiply": 1})


class TestCSECanNotReplace(unittest.TestCase, AssertOpCountEqualMixin):
    def test_can_not_replace_random(self):
        # Random OP will change the global state, it has side effect
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.rand([2, 2], dtype="float32")
            x2 = paddle.rand([2, 2], dtype="float32")

            self.assert_op_count_equal(main_program, {"pd_op.uniform": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.uniform": 2})

    def test_can_not_replace_print(self):
        # print op will output to stdout, it has side effect
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")

            paddle.static.Print(x1)
            paddle.static.Print(x1)

            self.assert_op_count_equal(main_program, {"pd_op.print": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.print": 2})

    def test_can_not_replace_used_by_inplace_op(self):
        # If the op results is used by inplace op, i.e. the results
        # will be modified, we can not replace it
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")

            a = x1 + x2
            b = x1 + x2
            paddle.assign(paddle.full([2, 2], 0, dtype="float32"), b)

            self.assert_op_count_equal(main_program, {"pd_op.add": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.add": 2})

    def test_can_not_replace_op_with_subblocks(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")
            cond = paddle.static.data("cond", [], dtype="bool")
            loop_var = paddle.static.data("i", [], dtype="int32")

            def loop_body(loop_var):
                tmp_var = x1 * x2
                return [loop_var + 1]

            def get_cond(loop_var):
                return cond

            paddle.static.nn.while_loop(get_cond, loop_body, [loop_var])
            paddle.static.nn.while_loop(get_cond, loop_body, [loop_var])

            self.assert_op_count_equal(main_program, {"pd_op.while": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.while": 2})


@unittest.skipUnless(
    paddle.is_compiled_with_cinn(),
    "This case only works when compiled with CINN",
)
class TestCSEDenyFullInCinn(unittest.TestCase, AssertOpCountEqualMixin):
    CINN_FLAGS_NAME = "FLAGS_use_cinn"

    def test_replace_full_without_cinn(self):
        with flag_guard(
            self.CINN_FLAGS_NAME, False
        ), program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.full([2], 1.0, dtype="float32")
            x2 = paddle.full([2], 1.0, dtype="float32")

            self.assert_op_count_equal(main_program, {"pd_op.full": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.full": 1})

    def test_replace_full_with_cinn(self):
        with flag_guard(
            self.CINN_FLAGS_NAME, True
        ), program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.full([2], 1.0, dtype="float32")
            x2 = paddle.full([2], 1.0, dtype="float32")

            self.assert_op_count_equal(main_program, {"pd_op.full": 2})
            paddle.base.libpaddle.pir.apply_cse_pass(main_program)
            self.assert_op_count_equal(main_program, {"pd_op.full": 2})


if __name__ == "__main__":
    unittest.main()
