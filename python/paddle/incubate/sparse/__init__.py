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

from .creation import sparse_coo_tensor
from .creation import sparse_csr_tensor

from .unary import sin
from .unary import tan
from .unary import asin
from .unary import atan
from .unary import sinh
from .unary import tanh
from .unary import asinh
from .unary import atanh
from .unary import sqrt
from .unary import square
from .unary import log1p
from .unary import abs
from .unary import pow
from .unary import cast
from .unary import neg
from .unary import coalesce
from .unary import deg2rad
from .unary import rad2deg
from .unary import expm1

from .binary import mv
from .binary import matmul
from .binary import masked_matmul
from .binary import add
from .binary import divide
from .binary import multiply
from .binary import subtract
from .binary import is_same_shape

from .multiary import addmm

from . import nn

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
    'cast',
    'neg',
    'deg2rad',
    'rad2deg',
    'expm1',
    'mv',
    'matmul',
    'masked_matmul',
    'addmm',
    'add',
    'subtract',
    'multiply',
    'divide',
    'coalesce',
    'is_same_shape',
]
