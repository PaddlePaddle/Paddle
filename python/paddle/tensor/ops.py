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
    acos_,
    acosh,
    acosh_,
    asin,
    asin_,
    asinh,
    asinh_,
    atan,
    atan_,
    atanh,
    atanh_,
    ceil,
    ceil_,
    cos,
    cos_,
    cosh,
    cosh_,
    erf,
    erf_,
    exp,
    exp_,
    expm1,
    expm1_,
    floor,
    floor_,
    reciprocal,
    reciprocal_,
    round,
    round_,
    rsqrt,
    rsqrt_,
    sigmoid,
    sigmoid_,
    sin,
    sin_,
    sinh,
    sinh_,
    sqrt,
    sqrt_,
    square,
    square_,
    tan,
    tan_,
)

from .layer_function_generator import generate_layer_fn

__all__ = []

# It is a hot fix in some unittest using:
#   paddle.scale(x=x, scale=10.0, out=out_var)
# e.g.: test_program_code.py, test_dist_train.py
globals()['_scale'] = generate_layer_fn('scale')
