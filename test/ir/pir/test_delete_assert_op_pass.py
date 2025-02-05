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


def apply_delete_assert_op_pass(program):
    pm = paddle.pir.PassManager(2)
    pm.add_pass("delete_assert_op_pass", {})
    pm.run(program)


class TestDeleteAssertOpBasic(unittest.TestCase, AssertOpCountEqualMixin):
    def test_basic(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="float32")
            x2 = paddle.static.data("x2", [2, 2], dtype="float32")

            # Computation
            cond = x1 == x2
            data = [x1, x2]
            summarize = 2
            paddle.static.nn.control_flow.Assert(cond, data, summarize)

            # ApplyPass
            self.assert_op_count_equal(
                main_program, {"pd_op.assert": 1, "builtin.combine": 1}
            )
            apply_delete_assert_op_pass(main_program)
            self.assert_op_count_equal(
                main_program, {"pd_op.assert": 0, "builtin.combine": 0}
            )

    def test_basic2(self):
        with program_scope_guard() as main_program:
            # Inputs
            x = paddle.static.data("x1", [], dtype="bool")

            # Computation
            cond = x
            data = []
            summarize = 0
            paddle.static.nn.control_flow.Assert(cond, data, summarize)

            self.assert_op_count_equal(
                main_program, {"pd_op.assert": 1, "builtin.combine": 1}
            )
            apply_delete_assert_op_pass(main_program)
            self.assert_op_count_equal(
                main_program, {"pd_op.assert": 0, "builtin.combine": 0}
            )


class TestDeleteAssertOpSubBlock(unittest.TestCase, AssertOpCountEqualMixin):
    def test_subblock(self):
        with program_scope_guard() as main_program:
            # Inputs
            x1 = paddle.static.data("x1", [2, 2], dtype="int32")
            x2 = paddle.static.data("x2", [2, 2], dtype="int32")
            cond = paddle.static.data("cond", [], dtype="bool")
            loop_var = paddle.static.data("i", [], dtype="int32")

            def computation(x1, x2):
                paddle.static.nn.control_flow.Assert(cond, [], 0)

            # Assert in global block
            computation(x1, x2)

            # Assert in subblock
            def loop_body(loop_var):
                computation(x1, x2)
                return [loop_var + 1]

            def get_cond(loop_var):
                return cond

            paddle.static.nn.while_loop(get_cond, loop_body, [loop_var])

            self.assert_op_count_equal(
                main_program, {"pd_op.assert": 2, "builtin.combine": 2}
            )
            apply_delete_assert_op_pass(main_program)
            self.assert_op_count_equal(
                main_program, {"pd_op.assert": 0, "builtin.combine": 0}
            )


if __name__ == "__main__":
    unittest.main()
