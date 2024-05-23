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

from typing import TYPE_CHECKING, Any, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from paddle import Tensor

Numberic: TypeAlias = Union[int, float, complex, np.number, "Tensor"]
TensorLike: TypeAlias = Union[npt.NDArray[Any], "Tensor", Numberic]

_T = TypeVar("_T", bound=Numberic)
_SeqLevel1: TypeAlias = Sequence[_T]
_SeqLevel2: TypeAlias = Sequence[Sequence[_T]]
_SeqLevel3: TypeAlias = Sequence[Sequence[Sequence[_T]]]
_SeqLevel4: TypeAlias = Sequence[Sequence[Sequence[Sequence[_T]]]]
_SeqLevel5: TypeAlias = Sequence[Sequence[Sequence[Sequence[Sequence[_T]]]]]
_SeqLevel6: TypeAlias = Sequence[
    Sequence[Sequence[Sequence[Sequence[Sequence[_T]]]]]
]

IntSequence: TypeAlias = _SeqLevel1[int]

NumbericSequence: TypeAlias = _SeqLevel1[Numberic]

NestedSequence: TypeAlias = Union[
    _T,
    _SeqLevel1[_T],
    _SeqLevel2[_T],
    _SeqLevel3[_T],
    _SeqLevel4[_T],
    _SeqLevel5[_T],
    _SeqLevel6[_T],
]

NestedNumbericSequence: TypeAlias = NestedSequence[Numberic]

TensorOrTensors: TypeAlias = Union["Tensor", Sequence["Tensor"]]
