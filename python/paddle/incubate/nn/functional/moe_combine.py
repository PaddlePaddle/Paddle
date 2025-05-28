# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING

from paddle import _C_ops

# from ....framework import LayerHelper, in_dynamic_or_pir_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor


def moe_combine(
    x: Tensor,
    combine_weights: Tensor,
    scatter_index: Tensor,
    name: str | None = None,
) -> Tensor:
    """
    Args:
        x: Input tensor [seq, dim]
        combine_weights: Combination weights [s, k]
        scatter_index: Scatter indices [k, s] dtype=int32

    Returns:
        Output Combined output [s, dim]
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.moe_combine(x, combine_weights, scatter_index)
    helper = LayerHelper('moe_combine', **locals())
    y = helper.create_variable_for_type_inference(dtype=x.dtype)
    inputs = {
        'x': x,
        'combine_weights': combine_weights,
        'scatter_index': scatter_index,
    }
    helper.append_op(type='moe_combine', inputs=inputs, outputs={'y': y})
    return y
