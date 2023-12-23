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
from paddle.tensor import (  # noqa: F401
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atanh,
    broadcast_shape,
    broadcast_to,
    concat,
    cos,
    cosh,
    cumprod,
    cumsum,
    digamma,
    divide,
    erf,
    erfinv,
    exp,
    expm1,
    fill_constant,
    full,
    gather,
    greater_equal,
    lgamma,
    log,
    log1p,
    logcumsumexp,
    logit,
    logsumexp,
    max,
    mean,
    min,
    multiply,
    ones,
    pow,
    prod,
    reshape,
    rsqrt,
    sign,
    sin,
    sinh,
    sqrt,
    subtract,
    sum,
    tan,
    tanh,
    tile,
    uniform,
    zeros,
)
from paddle.tensor.creation import assign, zeros_like  # noqa: F401
from paddle.tensor.manipulation import cast  # noqa: F401
from paddle.tensor.math import maximum, minimum  # noqa: F401

"""
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
    'maximum',
    'min',
    'minimum',
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
    'rsqrt',
]

others = [
    'assign',
    'broadcast_to',
    'cast',
    'fill_constant',
    'reshape',
    'gather'
    'full',
    'tile',
    'concat',
    'uniform',
    'greater_equal',
    'zeros_like',
    'transpose',
]
"""
