# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.layers.tensor import fill_constant  # noqa: F401
from paddle.tensor import abs  # noqa: F401
from paddle.tensor import acos  # noqa: F401
from paddle.tensor import acosh  # noqa: F401
from paddle.tensor import add  # noqa: F401
from paddle.tensor import asin  # noqa: F401
from paddle.tensor import asinh  # noqa: F401
from paddle.tensor import atan  # noqa: F401
from paddle.tensor import atanh  # noqa: F401
from paddle.tensor import broadcast_shape  # noqa: F401
from paddle.tensor import broadcast_to  # noqa: F401
from paddle.tensor import cos  # noqa: F401
from paddle.tensor import cosh  # noqa: F401
from paddle.tensor import cumprod  # noqa: F401
from paddle.tensor import cumsum  # noqa: F401
from paddle.tensor import digamma  # noqa: F401
from paddle.tensor import divide  # noqa: F401
from paddle.tensor import erf  # noqa: F401
from paddle.tensor import erfinv  # noqa: F401
from paddle.tensor import exp  # noqa: F401
from paddle.tensor import expm1  # noqa: F401
from paddle.tensor import full  # noqa: F401
from paddle.tensor import lgamma  # noqa: F401
from paddle.tensor import log  # noqa: F401
from paddle.tensor import log1p  # noqa: F401
from paddle.tensor import logcumsumexp  # noqa: F401
from paddle.tensor import logit  # noqa: F401
from paddle.tensor import logsumexp  # noqa: F401
from paddle.tensor import max  # noqa: F401
from paddle.tensor import mean  # noqa: F401
from paddle.tensor import min  # noqa: F401
from paddle.tensor import multiply  # noqa: F401
from paddle.tensor import ones  # noqa: F401
from paddle.tensor import pow  # noqa: F401
from paddle.tensor import prod  # noqa: F401
from paddle.tensor import reshape  # noqa: F401
from paddle.tensor import sign  # noqa: F401
from paddle.tensor import sin  # noqa: F401
from paddle.tensor import sinh  # noqa: F401
from paddle.tensor import sqrt  # noqa: F401
from paddle.tensor import subtract  # noqa: F401
from paddle.tensor import sum  # noqa: F401
from paddle.tensor import tan  # noqa: F401
from paddle.tensor import tanh  # noqa: F401
from paddle.tensor import zeros  # noqa: F401
from paddle.tensor.creation import assign  # noqa: F401
from paddle.tensor.manipulation import cast  # noqa: F401

math_op = [
    'add',
    'subtract',
    'multiply',
    'divide',
    'abs',
    'pow',
    'sign',
    'sum',
    'prod',
    'cumsum',
    'cumprod',
    'digamma',
    'lgamma',
    'erf',
    'erfinv',
    'exp',
    'expm1',
    'log',
    'log1p',
    'logsumexp',
    'logcumsumexp',
    'logit',
    'max',
    'min',
]

trigonometric_op = [
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'tanh',
    'asin',
    'acos',
    'atan',
    'asinh',
    'acosh',
    'atanh',
]

sub_prim = [
    'mean',
    'ones',
    'zeros',
    'sqrt',
]

others = [
    'assign',
    'broadcast_to',
    'cast',
    'fill_constant',
    'reshape',
    'full',
]

__all__ = []
__all__.extend(math_op)
__all__.extend(trigonometric_op)
__all__.extend(sub_prim)
__all__.extend(others)

__all__.sort()
