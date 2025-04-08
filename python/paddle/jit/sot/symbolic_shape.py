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
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:

    from .utils.magic_methods import BinaryOp, UnaryOp

_T = TypeVar("_T", "int", "float", "bool")


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


class SymbolicValue(Generic[_T]):
    example_value: _T | None

    def __init__(self, example_value=None):
        self.example_value = example_value

    def __repr__(self) -> str:
        if self.is_backed():
            return f"{self.__class__.__name__}({self.example_value})"
        return f"{self.__class__.__name__}()"

    def get_static_type(self) -> type[_T]:
        raise NotImplementedError("get_py_type is not implemented.")

    def is_backed(self):
        return self.example_value is not None

    def get_example_value(self) -> _T:
        if self.example_value is None:
            raise ValueError(f"{self} is not backed by a value.")
        return self.example_value


class SymbolicBool(SymbolicValue):
    def get_static_type(self) -> type[bool]:
        return bool


class SymbolicInt(SymbolicValue):
    def get_static_type(self) -> type[int]:
        return int


class SymbolicFloat(SymbolicValue):
    def get_static_type(self) -> type[float]:
        return float
