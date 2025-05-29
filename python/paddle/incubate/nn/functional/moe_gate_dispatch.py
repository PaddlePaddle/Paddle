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

import paddle
from paddle import _C_ops

# from ....framework import LayerHelper, in_dynamic_or_pir_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor


def moe_gate_dispatch(
    x: Tensor,
    gate_logits: Tensor,
    corr_bias: Tensor,
    k: int,
    capacity: int,
    use_pad: bool,
    name: str | None = None,
) -> Tensor:
    """
    Args:
        x:
        gate_logits:
        corr_bias:
        k:
        capacity:
        use_pad:

    Returns:
        y:
        combine_weights:
        scatter_index:
        expert_offset:
        expert_id:
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.moe_gate_dispatch(
            x, gate_logits, corr_bias, k, capacity, use_pad
        )

    helper = LayerHelper('moe_gate_dispatch', **locals())
    y = helper.create_variable_for_type_inference(dtype=x.dtype)
    combine_weights = helper.create_variable_for_type_inference(
        dtype=paddle.float32
    )
    scatter_index = helper.create_variable_for_type_inference(
        dtype=paddle.int32
    )
    expert_offset = helper.create_variable_for_type_inference(
        dtype=paddle.int64
    )
    expert_id = helper.create_variable_for_type_inference(dtype=paddle.int32)

    inputs = {
        'x': x,
        'gate_logits': gate_logits,
        'corr_bias': corr_bias,
    }
    attrs = {
        'k': k,
        'capacity': capacity,
        'use_pad': use_pad,
    }
    outputs = {
        'y': y,
        'combine_weights': combine_weights,
        'scatter_index': scatter_index,
        'expert_offset': expert_offset,
        'expert_id': expert_id,
    }
    helper.append_op(
        type='moe_gate_dispatch',
        inputs=inputs,
        attrs=attrs,
        outputs=outputs,
    )
    return y, combine_weights, scatter_index, expert_offset, expert_id
