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


def moe_gate_dispatch_permute(
    x: Tensor,
    gate_logits: Tensor,
    corr_bias: Tensor,
    k: int,
    capacity: int,
    world_size: int,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Dispatch and permute for Mixture of Experts (MoE).

    Args:
        x: Input tensor [batch_size, seq_len, hidden_dim].
        gate_logits: Gate logits for choosing experts [batch_size, seq_len, num_experts].
        corr_bias: Optional correction bias to adjust gate logits.
        k: Top-k experts to be selected.
        capacity: The maximum number of tokens an expert can handle.
        world_size: Number of distributed processes.
        name: Optional name for the operation.

    Returns:
        Tuple of Tensors containing:
        - y: Output tensor after dispatch and permute.
        - combine_weights: Weights for combining experts' outputs.
        - scatter_index: Indices for scattering inputs to experts.
        - expert_offset: Offset indices for each expert.
        - expert_id: IDs of selected experts for each position.
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.moe_gate_dispatch_permute(
            x, gate_logits, corr_bias, k, capacity, world_size
        )

    helper = LayerHelper('moe_gate_dispatch_permute', **locals())
    y = helper.create_variable_for_type_inference(dtype=x.dtype)
    combine_weights = helper.create_variable_for_type_inference(dtype='float')
    scatter_index = helper.create_variable_for_type_inference(dtype='int32')
    expert_offset = helper.create_variable_for_type_inference(dtype='int32')
    expert_id = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {
        'x': x,
        'gate_logits': gate_logits,
        'corr_bias': corr_bias if corr_bias is not None else None,
    }
    attrs = {'k': k, 'capacity': capacity, 'world_size': world_size}
    outputs = {
        'y': y,
        'combine_weights': combine_weights,
        'scatter_index': scatter_index,
        'expert_offset': expert_offset,
        'expert_id': expert_id,
    }

    helper.append_op(
        type='moe_gate_dispatch_permute',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs,
    )
    return y, combine_weights, scatter_index, expert_offset, expert_id
