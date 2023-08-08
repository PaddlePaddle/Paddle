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

from .primitives import *  # noqa: F403
from .register import register_decomp


@register_decomp('pd.mean')
def mean(x, axis, keepdim):
    """define composite rule of op mean"""
    x_shape = x.shape
    axes = axis or tuple(range(0, len(x_shape)))
    axes = (axes,) if isinstance(axes, int) else axes
    sum_x = sum(x, axis=axes, keepdim=keepdim)
    value_to_fill = 1
    for axis in axes:
        value_to_fill *= x_shape[axis]
    norm = fill_constant(
        shape=[],
        value=value_to_fill,
        dtype=sum_x.dtype,
    )
    res = divide(sum_x, norm)
    return res
