# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""The postprocess list of a slice ref, and the only place that parses it.

A slice ref says "take this region of a source key, run its postprocess list
on it, and put the result at this region of the destination". The list is
ordered and holds three kinds of string entries:

* a permute, ``"[1, 0]"`` -- transpose the data;
* a reshape, ``"reshape:[2, 3]"`` -- reshape the data to this local shape;
* a cast, a dtype name such as ``"float16"``.

The plan layer (``aoa_engine``) writes these lists and the copy layer
(``dcp.utils``) replays them; both go through the helpers below instead of
looking at the strings themselves.
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_RESHAPE_PREFIX = "reshape:"


def is_permute(operation: object) -> bool:
    return isinstance(operation, str) and operation.startswith("[")


def is_reshape(operation: object) -> bool:
    return isinstance(operation, str) and operation.startswith(_RESHAPE_PREFIX)


def make_reshape_op(shape: Sequence[int]) -> str:
    """The entry that reshapes the data it travels with into ``shape``.

    ``shape`` is the local shape of that data, not a global one, so the entry
    stays valid wherever later statements move the slice ref.
    """
    return f"{_RESHAPE_PREFIX}{list(shape)}"


def reshape_op_shape(operation: str) -> tuple[int, ...]:
    """The shape a reshape entry reshapes into."""
    assert is_reshape(operation), f"{operation!r} is not a reshape entry"
    return tuple(ast.literal_eval(operation[len(_RESHAPE_PREFIX) :]))


def shape_after_postprocess(
    shape: Sequence[int], postprocess_list: list[str] | None
) -> list[int]:
    """The shape a slice of ``shape`` has once its postprocess list has run."""
    shape = list(shape)
    for operation in postprocess_list or []:
        if is_permute(operation):
            shape = transpose_list(shape, ast.literal_eval(operation))
        elif is_reshape(operation):
            target = reshape_op_shape(operation)
            if math.prod(shape) != math.prod(target):
                raise ValueError(
                    f"{operation} cannot apply to a slice of shape "
                    f"{tuple(shape)}: the element counts differ."
                )
            shape = list(target)
    return shape


def transpose_list(
    li: list | tuple, permutation: Sequence[int]
) -> list | tuple:
    """``li`` reordered by ``permutation``, as the same kind of sequence."""
    trans_list = []
    for idx in permutation:
        trans_list.append(li[idx])
    if isinstance(li, tuple):
        return tuple(trans_list)
    else:
        return trans_list


def invert_permutation(p: Sequence[int]) -> list[int]:
    q = [0] * len(p)
    for i, pi in enumerate(p):
        q[pi] = i
    return q
