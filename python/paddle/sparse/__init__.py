#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from . import nn  # noqa: F401
from .binary import (
    add,
    divide,
    is_same_shape,
    mask_as,
    masked_matmul,
    matmul,
    multiply,
    mv,
    subtract,
)
from .creation import sparse_coo_tensor, sparse_csr_tensor
from .multiary import addmm
from .unary import (
    abs,
    asin,
    asinh,
    atan,
    atanh,
    cast,
    coalesce,
    deg2rad,
    expm1,
    isnan,
    log1p,
    neg,
    pca_lowrank,
    pow,
    rad2deg,
    reshape,
    sin,
    sinh,
    slice,
    sqrt,
    square,
    sum,
    tan,
    tanh,
    transpose,
)

__all__ = [
    'sparse_coo_tensor',
    'sparse_csr_tensor',
    'sin',
    'tan',
    'asin',
    'atan',
    'sinh',
    'tanh',
    'asinh',
    'atanh',
    'sqrt',
    'square',
    'log1p',
    'abs',
    'pow',
    'pca_lowrank',
    'cast',
    'neg',
    'deg2rad',
    'rad2deg',
    'expm1',
    'mv',
    'matmul',
    'mask_as',
    'masked_matmul',
    'addmm',
    'add',
    'subtract',
    'transpose',
    'sum',
    'multiply',
    'divide',
    'coalesce',
    'is_same_shape',
    'reshape',
    'isnan',
    'slice',
]
