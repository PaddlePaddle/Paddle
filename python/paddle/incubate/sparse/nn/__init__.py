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

<<<<<<< HEAD:python/paddle/incubate/sparse/nn/__init__.py
from . import functional

from .layer.activation import ReLU
from .layer.norm import BatchNorm, SyncBatchNorm
from .layer.activation import Softmax
from .layer.activation import ReLU6
from .layer.activation import LeakyReLU
from .layer.conv import Conv3D
from .layer.conv import SubmConv3D
from .layer.pooling import MaxPool3D

__all__ = [
    'ReLU',
    'ReLU6',
    'LeakyReLU',
    'Softmax',
    'BatchNorm',
    'SyncBatchNorm',
    'Conv3D',
    'SubmConv3D',
    'MaxPool3D',
=======
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
from .unary import transpose
from .unary import reshape

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
    'transpose',
    'multiply',
    'divide',
    'coalesce',
    'is_same_shape',
    'reshape',
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f:python/paddle/sparse/__init__.py
]
