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


# TODO(kevincheng2): python implementation of prim feature,
# now it has been sunk to c++, waiting for further deletion.
@register_decomp('pd_op.unsqueeze')
def unsqueeze(x, axis):
    """define composite rule of op unsqueeze"""
    """using reshape to implement unsqueeze op"""
    axis = axis.get_defining_op().attrs()["value"]
    x_shape = list(x.shape)
    axis_list = list(axis)
    for i in axis_list:
        if i < 0:
            i += len(x_shape) + 1
        x_shape = (
            x_shape[:i]
            + [
                1,
            ]
            + x_shape[i:]
        )
    out = reshape(x, x_shape)
    return [out, None]
