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

from typing import TYPE_CHECKING

import paddle

from ..utils.exceptions import InnerError

if TYPE_CHECKING:
    from ..opcode_translator.executor.guard import StringifiedExpression


class ConstraintNode:
    def __init__(self, inputs: list[ConstraintNode]):
        self.inputs = inputs

    def create_guard_expr(
        self, extern_vars: dict[str, StringifiedExpression]
    ) -> StringifiedExpression:
        raise NotImplementedError

    def create_guard_node(
        self, extern_vars: dict[str, paddle.framework.core.ExprNodeBase]
    ) -> paddle.framework.core.ExprNodeBase:
        raise NotImplementedError(
            f"{self.__class__.__name__}.create_guard_node is not implemented"
        )


class LeafConstraintNode(ConstraintNode):
    def __init__(self):
        super().__init__([])


class UnaryConstraintNode(ConstraintNode):
    READABLE_SYMBOL: str

    def __init__(self, input: ConstraintNode):
        super().__init__([input])
        self.input = input

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.input})"

    def create_guard_expr(
        self, extern_vars: dict[str, StringifiedExpression]
    ) -> StringifiedExpression:
        from ..opcode_translator.executor.guard import (
            StringifiedExpression,
            union_free_vars,
        )

        input = self.input.create_guard_expr(extern_vars)
        return StringifiedExpression(
            f"{self.READABLE_SYMBOL}({{}})",
            [input],
            union_free_vars(input.free_vars),
        )

    def create_guard_node(
        self, extern_vars: dict[str, paddle.framework.core.ExprNodeBase]
    ) -> paddle.framework.core.ExprNodeBase:
        input = self.input.create_guard_node(extern_vars)
        return paddle.framework.core.UnaryExprNode(input, self.READABLE_SYMBOL)


class BinaryConstraintNode(ConstraintNode):
    READABLE_SYMBOL: str

    def __init__(self, lhs: ConstraintNode, rhs: ConstraintNode):
        super().__init__([lhs, rhs])
        self.lhs = lhs
        self.rhs = rhs

    def create_guard_expr(
        self, extern_vars: dict[str, StringifiedExpression]
    ) -> StringifiedExpression:
        from ..opcode_translator.executor.guard import (
            StringifiedExpression,
            union_free_vars,
        )

        lhs = self.lhs.create_guard_expr(extern_vars)
        rhs = self.rhs.create_guard_expr(extern_vars)
        return StringifiedExpression(
            f"({{}} {self.READABLE_SYMBOL} {{}})",
            [lhs, rhs],
            union_free_vars(lhs.free_vars, rhs.free_vars),
        )

    def create_guard_node(
        self, extern_vars: dict[str, paddle.framework.core.ExprNodeBase]
    ) -> paddle.framework.core.ExprNodeBase:
        lhs = self.lhs.create_guard_node(extern_vars)
        rhs = self.rhs.create_guard_node(extern_vars)
        return paddle.framework.core.BinaryExprNode(
            lhs, rhs, self.READABLE_SYMBOL
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.lhs}, {self.rhs})"


class ConstantConstraintNode(LeafConstraintNode):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def create_guard_expr(
        self, extern_vars: dict[str, StringifiedExpression]
    ) -> StringifiedExpression:
        from ..opcode_translator.executor.guard import (
            StringifiedExpression,
        )

        return StringifiedExpression(f"{self.value!r}", [], {})

    def create_guard_node(
        self, extern_vars: dict[str, paddle.framework.core.ExprNodeBase]
    ) -> paddle.framework.core.ExprNodeBase:
        return paddle.framework.core.ConstantExprNode(self.value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class SymbolicConstraintNode(LeafConstraintNode):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def create_guard_expr(
        self, extern_vars: dict[str, StringifiedExpression]
    ) -> StringifiedExpression:
        from ..opcode_translator.executor.guard import (
            StringifiedExpression,
            union_free_vars,
        )

        if self.name not in extern_vars:
            raise InnerError(
                f"Symbolic variable {self.name} not found in extern_vars."
            )
        return StringifiedExpression(
            "{}",
            [extern_vars[self.name]],
            union_free_vars(extern_vars[self.name].free_vars),
        )

    def create_guard_node(
        self, extern_vars: dict[str, paddle.framework.core.ExprNodeBase]
    ) -> paddle.framework.core.ExprNodeBase:
        if self.name not in extern_vars:
            raise InnerError(
                f"Symbolic variable {self.name} not found in extern_vars."
            )
        return extern_vars[self.name]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class NegativeConstraintNode(UnaryConstraintNode):
    READABLE_SYMBOL = "-"


class BitwiseNotConstraintNode(UnaryConstraintNode):
    READABLE_SYMBOL = "~"


class AddConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "+"


class SubConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "-"


class MulConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "*"


class TrueDivConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "/"


class FloorDivConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "//"


class ModConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "%"


class PowConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "**"


class BitwiseLShiftConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "<<"


class BitwiseRShiftConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = ">>"


class BitwiseAndConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "&"


class BitwiseOrConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "|"


class BitwiseXorConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "^"


class LogicalToBoolConstraintNode(UnaryConstraintNode):
    READABLE_SYMBOL = "bool"


class LogicalNotConstraintNode(UnaryConstraintNode):
    READABLE_SYMBOL = "not"


class EqualConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "=="


class NotEqualConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "!="


class LessThanConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "<"


class LessEqualConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = "<="


class GreaterThanConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = ">"


class GreaterEqualConstraintNode(BinaryConstraintNode):
    READABLE_SYMBOL = ">="
