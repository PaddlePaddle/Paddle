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

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .utils import hashable

if TYPE_CHECKING:
    BinaryOp = Callable[[Any, Any], Any]
    UnaryOp = Callable[[Any], Any]


INPLACE_BINARY_OPS_TO_MAGIC_NAMES: dict[BinaryOp, tuple[str, BinaryOp]] = {
    # inplace op fn: (magic name, non-inplace op fn)
    operator.iadd: ("__iadd__", operator.add),
    operator.iand: ("__iand__", operator.and_),
    operator.iconcat: ("__iconcat__", operator.concat),
    operator.ifloordiv: ("__ifloordiv__", operator.floordiv),
    operator.ilshift: ("__ilshift__", operator.lshift),
    operator.imatmul: ("__imatmul__", operator.matmul),
    operator.imod: ("__imod__", operator.mod),
    operator.imul: ("__imul__", operator.mul),
    operator.ior: ("__ior__", operator.or_),
    operator.ipow: ("__ipow__", operator.pow),
    operator.irshift: ("__irshift__", operator.rshift),
    operator.isub: ("__isub__", operator.sub),
    operator.itruediv: ("__itruediv__", operator.truediv),
    operator.ixor: ("__ixor__", operator.xor),
}

NON_INPLACE_BINARY_OPS_TO_MAGIC_NAMES: dict[
    BinaryOp, tuple[str, str | None]
] = {
    # op fn: (magic name, reverse magic name)
    operator.add: ("__add__", "__radd__"),
    operator.and_: ("__and__", "__rand__"),
    operator.contains: ("__contains__", None),
    operator.delitem: ("__delitem__", None),
    operator.eq: ("__eq__", "__eq__"),
    operator.floordiv: ("__floordiv__", "__rfloordiv__"),
    operator.ge: ("__ge__", "__le__"),
    operator.getitem: ("__getitem__", None),
    operator.gt: ("__gt__", "__lt__"),
    operator.le: ("__le__", "__ge__"),
    operator.lshift: ("__lshift__", "__rlshift__"),
    operator.lt: ("__lt__", "__gt__"),
    operator.matmul: ("__matmul__", "__rmatmul__"),
    operator.mod: ("__mod__", "__rmod__"),
    operator.mul: ("__mul__", "__rmul__"),
    operator.ne: ("__ne__", "__ne__"),
    operator.or_: ("__or__", "__ror__"),
    operator.pow: ("__pow__", "__rpow__"),
    operator.rshift: ("__rshift__", "__rrshift__"),
    operator.sub: ("__sub__", "__rsub__"),
    operator.truediv: ("__truediv__", "__rtruediv__"),
    operator.xor: ("__xor__", "__rxor__"),
}

UNARY_OPS_TO_MAGIC_NAMES: dict[UnaryOp, str] = {
    operator.neg: "__neg__",
    operator.invert: "__invert__",
    operator.pos: "__pos__",
    operator.abs: "__abs__",
    operator.index: "__index__",
    operator.inv: "__inv__",
    operator.invert: "__invert__",
    operator.not_: "__not__",
    operator.pos: "__pos__",
    operator.truth: "__bool__",
    bool: "__bool__",
    abs: "__abs__",
    float: "__float__",
    len: "__len__",
    int: "__int__",
}
# TODO(SigureMo): support any, all, sum


INPLACE_BINARY_OPS = set(INPLACE_BINARY_OPS_TO_MAGIC_NAMES.keys())
NON_INPLACE_BINARY_OPS = set(NON_INPLACE_BINARY_OPS_TO_MAGIC_NAMES.keys())
BINARY_OPS = INPLACE_BINARY_OPS | NON_INPLACE_BINARY_OPS
UNARY_OPS = set(UNARY_OPS_TO_MAGIC_NAMES.keys())


@dataclass
class MagicMethod:
    name: str
    is_inplace: bool = False
    is_reverse: bool = False


def magic_method_builtin_dispatch(fn: BinaryOp | UnaryOp) -> list[MagicMethod]:
    if not hashable(fn):
        return []
    if fn in INPLACE_BINARY_OPS:
        inplace_magic_name, non_inplace_op = INPLACE_BINARY_OPS_TO_MAGIC_NAMES[
            fn
        ]
        return [
            MagicMethod(inplace_magic_name, is_inplace=True)
        ] + magic_method_builtin_dispatch(non_inplace_op)
    elif fn in NON_INPLACE_BINARY_OPS:
        magic_name, reverse_magic_name = NON_INPLACE_BINARY_OPS_TO_MAGIC_NAMES[
            fn
        ]
        magic_methods = [MagicMethod(magic_name)]
        if reverse_magic_name is not None:
            magic_methods.append(
                MagicMethod(reverse_magic_name, is_reverse=True)
            )
        return magic_methods
    elif fn in UNARY_OPS:
        magic_name = UNARY_OPS_TO_MAGIC_NAMES[fn]
        return [MagicMethod(magic_name)]
    return []
