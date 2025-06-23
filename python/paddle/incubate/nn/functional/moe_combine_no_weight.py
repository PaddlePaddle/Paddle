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
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor


def moe_combine_no_weight(
    x: Tensor,
    scatter_index: Tensor,
    name: str | None = None,
) -> Tensor:
    """
    Args:
        x: Input tensor [num_tokens, hidden_size]
        scatter_index: Scatter indices [seq_len, k] dtype=int32

    Returns:
        Output Combined output [seq_len, hidden_size]
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.moe_combine_no_weight(x, scatter_index)
    helper = LayerHelper('moe_combine_no_weight', **locals())
    y = helper.create_variable_for_type_inference(dtype=x.dtype)
    inputs = {
        'x': x,
        'scatter_index': scatter_index,
    }
    helper.append_op(
        type='moe_combine_no_weight', inputs=inputs, outputs={'y': y}
    )
    return y
