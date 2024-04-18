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

from typing import List, Optional, Tuple, Union

from typing_extensions import TypeAlias

from .. import Tensor

DynamicShapeLike: TypeAlias = Union[
    Tuple[Optional[int]], list[Optional[int]], Tensor
]

# Note: Do not confrom to predefined naming style in pylint.
Shape0D: TypeAlias = Tuple[None]
Shape1D: TypeAlias = Tuple[int]
Shape2D: TypeAlias = Tuple[int, int]
Shape3D: TypeAlias = Tuple[int, int, int]
Shape4D: TypeAlias = Tuple[int, int, int, int]
Shape5D: TypeAlias = Tuple[int, int, int, int, int]
Shape6D: TypeAlias = Tuple[int, int, int, int, int, int]
ShapeND: TypeAlias = Tuple[int, ...]


ShapeLike: TypeAlias = Union[
    Shape0D,
    Shape1D,
    Shape2D,
    Shape3D,
    Shape4D,
    Shape5D,
    Shape6D,
    List[int],
    Tensor,
]


# for size parameters, eg, kernel_size, stride ...
Size1: TypeAlias = Union[int, Shape1D]
Size2: TypeAlias = Union[int, Shape2D]
Size3: TypeAlias = Union[int, Shape3D]
Size4: TypeAlias = Union[int, Shape4D]
Size5: TypeAlias = Union[int, Shape5D]
Size6: TypeAlias = Union[int, Shape6D]
SizeN: TypeAlias = Union[int, ShapeND]
