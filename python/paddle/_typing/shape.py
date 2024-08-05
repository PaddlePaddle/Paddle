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

from typing import TYPE_CHECKING, List, Sequence, Tuple, Union

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from .. import Tensor

_DynamicShapeLike: TypeAlias = Union[
    Sequence[Union[int, "Tensor", None]],
    "Tensor",
]


_StaticShapeLike: TypeAlias = Union[
    Sequence[int],
    "Tensor",
]

ShapeLike: TypeAlias = Union[_DynamicShapeLike, _StaticShapeLike]

# for size parameters, eg, kernel_size, stride ...
Size1: TypeAlias = Union[int, Tuple[int], List[int]]
Size2: TypeAlias = Union[int, Tuple[int, int], List[int]]
Size3: TypeAlias = Union[int, Tuple[int, int, int], List[int]]
Size4: TypeAlias = Union[int, Tuple[int, int, int, int], List[int]]
Size5: TypeAlias = Union[int, Tuple[int, int, int, int, int], List[int]]
Size6: TypeAlias = Union[int, Tuple[int, int, int, int, int, int], List[int]]
SizeN: TypeAlias = Union[int, Tuple[int, ...], List[int]]
