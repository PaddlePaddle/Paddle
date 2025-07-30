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

import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.magic_methods import BinaryOp, UnaryOp


def symbolic_to_bool(x):
    # Unified api for python number and paddle Tensor
    return x != 0


def symbolic_not(x):
    return x == 0


# All symbolic operations need unified for python number and paddle Tensor
SYMBOLIC_UNARY_MATH_OPS: list[UnaryOp] = [
    # Basic
    operator.neg,
    # Bitwise
    operator.invert,
]
SYMBOLIC_BINARY_MATH_OPS: list[BinaryOp] = [
    # Basic
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    # Bitwise
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.or_,
    operator.xor,
]
SYMBOLIC_UNARY_LOGICAL_OPS: list[UnaryOp] = [
    symbolic_to_bool,
    symbolic_not,
]
SYMBOLIC_BINARY_LOGICAL_OPS: list[BinaryOp] = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]
SYMBOLIC_MATH_OPS = SYMBOLIC_UNARY_MATH_OPS + SYMBOLIC_BINARY_MATH_OPS
SYMBOLIC_MATH_OPS = SYMBOLIC_UNARY_MATH_OPS + SYMBOLIC_BINARY_MATH_OPS
SYMBOLIC_UNARY_OPS: list[UnaryOp] = (
    SYMBOLIC_UNARY_MATH_OPS + SYMBOLIC_UNARY_LOGICAL_OPS
)
SYMBOLIC_BINARY_OPS: list[BinaryOp] = (
    SYMBOLIC_BINARY_MATH_OPS + SYMBOLIC_BINARY_LOGICAL_OPS
)
