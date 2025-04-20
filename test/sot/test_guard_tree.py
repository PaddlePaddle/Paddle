# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Callable

import paddle
from paddle.jit.sot.opcode_translator.custom_code import CustomCode

if TYPE_CHECKING:
    import types

z = paddle.to_tensor(1)


def guard_tree(
    callback: Callable[[types.FrameType], CustomCode],
):
    def foo(x, y):
        return x, y, z

    def inner(x, y):
        paddle.framework.core.set_eval_frame(callback)
        try:
            return foo(x, y)
        finally:
            paddle.framework.core.set_eval_frame(None)

    return inner


class TestGuardTree(unittest.TestCase):
    def test_guard_tree(self):
        def callback(frame, **kwargs):
            guard_int = paddle.framework.core.TypeMatchGuard(int)
            guard_tensor = paddle.framework.core.TypeMatchGuard(
                type(paddle.to_tensor(1))
            )
            guard_tensor_shape_eq = paddle.framework.core.ValueMatchGuard(
                paddle.to_tensor([1]).shape
            )
            guard_eq = paddle.framework.core.ValueMatchGuard(1)

            node_expr_x = paddle.framework.core.LocalVarExprNode("x")
            node_expr_y = paddle.framework.core.LocalVarExprNode("y")
            node_expr_z = paddle.framework.core.GlobalVarExprNode("z")
            node_expr_y_shape = paddle.framework.core.AttributeExprNode(
                node_expr_y, "shape"
            )
            node_expr_const_1 = paddle.framework.core.ConstantExprNode(1)

            node_guard_x_type_match_int = paddle.framework.core.GuardNode(
                guard_int, [node_expr_x]
            )
            node_guard_x_eq_1 = paddle.framework.core.GuardNode(
                guard_eq, [node_expr_x]
            )
            node_guard_y_type_match_int = paddle.framework.core.GuardNode(
                guard_int, [node_expr_y]
            )
            node_guard_y_type_match_tensor = paddle.framework.core.GuardNode(
                guard_tensor, [node_expr_y]
            )
            node_guard_y_tensor_shape = paddle.framework.core.GuardNode(
                guard_tensor_shape_eq, [node_expr_y_shape]
            )
            node_guard_z_type_match_tensor = paddle.framework.core.GuardNode(
                guard_tensor, [node_expr_z]
            )
            node_guard_1_eq_1 = paddle.framework.core.GuardNode(
                guard_eq, [node_expr_const_1]
            )

            guard_tree = paddle.framework.core.GuardTree(
                [
                    [node_guard_x_eq_1],
                    [node_guard_x_type_match_int, node_guard_y_type_match_int],
                    [
                        node_guard_x_type_match_int,
                        node_guard_y_type_match_tensor,
                        node_guard_y_tensor_shape,
                    ],
                    [node_guard_z_type_match_tensor],
                    [node_guard_1_eq_1],
                ]
            )
            self.assertEqual(guard_tree.lookup(frame), target)
            return CustomCode(frame.f_code, False)

        global z
        foo = guard_tree(callback)
        target = 0
        foo(1, 1)
        target = 1
        foo(2, 2)
        target = 2
        foo(2, paddle.to_tensor([2]))
        target = 3
        foo(2, 1.0)
        z = 1
        target = 4
        foo(2, 1.0)


if __name__ == "__main__":
    unittest.main()
