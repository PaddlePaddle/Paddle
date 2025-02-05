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
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
from typing_extensions import Never, TypeAlias

from .backport import EllipsisType

if TYPE_CHECKING:
    from paddle import ParamAttr, Tensor
    from paddle.nn.initializer import Initializer
    from paddle.regularizer import WeightDecayRegularizer


Numeric: TypeAlias = Union[int, float, complex, np.number, "Tensor"]
TensorLike: TypeAlias = Union[npt.NDArray[Any], "Tensor", Numeric]
_TensorIndexItem: TypeAlias = Union[
    None, bool, int, slice, "Tensor", EllipsisType
]
TensorIndex: TypeAlias = Union[
    _TensorIndexItem,
    Tuple[_TensorIndexItem, ...],
    List[_TensorIndexItem],
]


_T = TypeVar("_T")

NestedSequence = Union[_T, Sequence["NestedSequence[_T]"]]
NestedList = Union[_T, List["NestedList[_T]"]]
NestedStructure = Union[
    _T, Dict[str, "NestedStructure[_T]"], Sequence["NestedStructure[_T]"]
]
NumericSequence = Sequence[Numeric]
NestedNumericSequence: TypeAlias = NestedSequence[Numeric]
TensorOrTensors: TypeAlias = Union["Tensor", Sequence["Tensor"]]

ParamAttrLike: TypeAlias = Union[
    "ParamAttr", "Initializer", "WeightDecayRegularizer", str, bool
]


def unreached() -> Never:
    """Mark a code path as unreachable.
    Refer to https://typing.readthedocs.io/en/latest/source/unreachable.html#marking-code-as-unreachable
    """
    raise RuntimeError("Unreachable code path")
