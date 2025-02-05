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

# Backport
from .backport import EllipsisType as EllipsisType

# Basic
from .basic import (
    NestedList as NestedList,
    NestedNumericSequence as NestedNumericSequence,
    NestedSequence as NestedSequence,
    NestedStructure as NestedStructure,
    Numeric as Numeric,
    NumericSequence as NumericSequence,
    ParamAttrLike as ParamAttrLike,
    TensorIndex as TensorIndex,
    TensorLike as TensorLike,
    TensorOrTensors as TensorOrTensors,
    unreached as unreached,
)

# Device
from .device_like import (
    PlaceLike as PlaceLike,
)

# DType
from .dtype_like import DTypeLike as DTypeLike

# DataLayout
from .layout import (
    DataLayout0D as DataLayout0D,
    DataLayout1D as DataLayout1D,
    DataLayout1DVariant as DataLayout1DVariant,
    DataLayout2D as DataLayout2D,
    DataLayout3D as DataLayout3D,
    DataLayoutImage as DataLayoutImage,
    DataLayoutND as DataLayoutND,
)

# Shape
from .shape import (
    ShapeLike as ShapeLike,
    Size1 as Size1,
    Size2 as Size2,
    Size3 as Size3,
    Size4 as Size4,
    Size5 as Size5,
    Size6 as Size6,
    SizeN as SizeN,
)
