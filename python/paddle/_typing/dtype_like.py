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

from typing import TYPE_CHECKING, Literal, Type, Union

import numpy as np
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from paddle import dtype

_DTypeLiteral: TypeAlias = Literal[
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "complex64",
    "complex128",
    "bool",
]

_DTypeNumpy: TypeAlias = Union[
    Type[
        Union[
            np.uint8,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
            np.bool_,
        ]
    ],
    np.dtype,
]


DTypeLike: TypeAlias = Union["dtype", _DTypeNumpy, _DTypeLiteral]
