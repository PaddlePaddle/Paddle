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
from __future__ import annotations

from paddle._C_ops import (  # noqa: F401
    abs,
    abs_,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    ceil,
    cos,
    cosh,
    erf,
    erf_,
    exp,
    expm1,
    floor,
    reciprocal,
    round,
    round_,
    rsqrt,
    sigmoid,
    sin,
    sinh,
    sqrt,
    square,
    tan,
)
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from .layer_function_generator import (
    generate_inplace_fn,
    generate_layer_fn,
)

__inplace_unary_func__ = [
    'exp_',
    'sqrt_',
    'rsqrt_',
    'ceil_',
    'floor_',
    'reciprocal_',
    'sigmoid_',
    'sin_',
    'sinh_',
    'asin_',
    'asinh_',
    'cos_',
    'cosh_',
    'acos_',
    'acosh_',
    'tan_',
    'atan_',
    'atanh_',
    'expm1_',
    'square_',
]

__all__ = []

# It is a hot fix in some unittest using:
#   paddle.scale(x=x, scale=10.0, out=out_var)
# e.g.: test_program_code.py, test_dist_train.py
globals()['_scale'] = generate_layer_fn('scale')

for _OP in set(__inplace_unary_func__):
    func = generate_inplace_fn(_OP)
    func.__module__ = __name__
    _func = inplace_apis_in_dygraph_only(func)
    globals()[_OP] = _func
