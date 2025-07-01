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
        if paddle.device.is_compiled_with_custom_device('npu'):
            return math_moe_gate_dispatch(
                x, gate_logits, corr_bias, k, capacity, use_pad
            )
        else:
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


def topk_gating_softmax(
    gate_logits,
    corr_bias,
    topk,
):
    # Calculate scores with bias added (used for Top-K selection)
    scores_for_selection = (
        gate_logits + corr_bias if corr_bias is not None else gate_logits
    )
    # Get Top-K indices
    combine_weights, expert_id = paddle.topk(
        scores_for_selection, k=topk, axis=1
    )

    # Initialize source_rows: for each column, increment by 1 across rows,
    # then move to the next column after finishing one full column
    source_rows = paddle.to_tensor(
        [
            k_idx * gate_logits.shape[0] + row_idx
            for row_idx in range(gate_logits.shape[0])
            for k_idx in range(topk)
        ]
    )
    return combine_weights, expert_id, source_rows


def sorter_kernel(expert_id, source_rows):
    # Flatten all data
    flat_expert = expert_id.flatten()

    # Global sorting index (sorted by expert_id in ascending order)
    sort_idx = paddle.argsort(flat_expert)

    # Apply the sorting
    sorted_expert = paddle.gather(flat_expert, sort_idx)
    sorted_source = paddle.gather(source_rows, sort_idx)

    # Reshape back to [num_rows, k]
    return (sorted_expert.reshape(expert_id.shape), sorted_source)


def compute_total_rows_before_expert(permuted_experts, num_experts):
    expert_offset = paddle.searchsorted(
        permuted_experts.flatten(), paddle.arange(num_experts), right=True
    )
    return expert_offset


def initialize_moe_routing_matrix(
    unpermuted_input,
    gate_logits,
    expanded_dest_row_to_expanded_source_row,
    permuted_experts,
    expert_offset,
    combine_weights,
    capacity,
    use_pad=False,
):
    splits = paddle.concat(
        [
            paddle.to_tensor([0]),
            expert_offset,
            paddle.to_tensor([len(expanded_dest_row_to_expanded_source_row)]),
        ]
    )
    expanded_dest_row_to_expanded_source_row = paddle.concat(
        [
            paddle.sort(
                expanded_dest_row_to_expanded_source_row[
                    splits[i] : splits[i + 1]
                ]
            )
            for i in range(len(splits) - 1)
        ]
    )

    expanded_source_row_to_expanded_dest_row = paddle.scatter_nd(
        index=expanded_dest_row_to_expanded_source_row.unsqueeze(1),
        updates=paddle.arange(
            expanded_dest_row_to_expanded_source_row.shape[0]
        ),
        shape=[expanded_dest_row_to_expanded_source_row.shape[0]],
    )

    y = paddle.zeros(
        [gate_logits.shape[1] * capacity, unpermuted_input.shape[1]],
        dtype=unpermuted_input.dtype,
    )

    if use_pad:
        iexpert = paddle.gather(
            permuted_experts.flatten(),
            expanded_source_row_to_expanded_dest_row.flatten(),
        )

        extended_offset = paddle.concat(
            [paddle.zeros([1], dtype='int64'), expert_offset]
        )
        offset = paddle.gather(extended_offset, iexpert)

        iexpert_cap = iexpert * capacity
        row_in_expert = (
            expanded_source_row_to_expanded_dest_row.flatten() - offset
        )

        input_indices = (
            paddle.arange(row_in_expert.shape[0]) % unpermuted_input.shape[0]
        )

        y = paddle.scatter(
            x=y,
            index=row_in_expert + iexpert_cap,
            updates=unpermuted_input[input_indices],
            overwrite=True,
        )

        expanded_source_row_to_expanded_dest_row = (
            expanded_source_row_to_expanded_dest_row
            + iexpert_cap.reshape(
                expanded_source_row_to_expanded_dest_row.shape
            )
            - offset.reshape(expanded_source_row_to_expanded_dest_row.shape)
        )

        expanded_source_row_to_expanded_dest_row = (
            expanded_source_row_to_expanded_dest_row.reshape(
                [combine_weights.shape[1], combine_weights.shape[0]]
            )
        )
        mask = (
            row_in_expert.reshape(
                [combine_weights.shape[1], combine_weights.shape[0]]
            )
            < capacity
        )

        expanded_source_row_to_expanded_dest_row = paddle.where(
            mask,
            expanded_source_row_to_expanded_dest_row,
            paddle.zeros_like(expanded_source_row_to_expanded_dest_row),
        )
        combine_weights = paddle.where(
            mask.T, combine_weights, paddle.zeros_like(combine_weights)
        )

    return y, expanded_source_row_to_expanded_dest_row, combine_weights


def math_moe_gate_dispatch(x, gate_logits, corr_bias, k, capacity, use_pad):
    combine_weights, expert_id, source_rows = topk_gating_softmax(
        gate_logits, corr_bias, k
    )

    permuted_experts, permuted_rows = sorter_kernel(expert_id, source_rows)

    expert_offset = compute_total_rows_before_expert(
        permuted_experts, gate_logits.shape[1]
    )

    y, scatter_index, combine_weights = initialize_moe_routing_matrix(
        x,
        gate_logits,
        permuted_rows,
        permuted_experts,
        expert_offset,
        combine_weights,
        capacity,
        use_pad,
    )

    return y, combine_weights, scatter_index, expert_offset, expert_id
