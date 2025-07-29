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

if TYPE_CHECKING:
    from paddle import Tensor


def tokens_unzip_gather(
    hidden_states: Tensor,
    scale: Tensor | None,
    zipped_expertwise_rowmap: Tensor,
    expert_id: int,
    tokens_per_expert: list,
    padding_alignment: int,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Gather tokens and indices from specific expert in MoE training scenarios.

    Args:
        hidden_states (Tensor): The input tensor containing tokens to be permuted, stored in row-major layout.
            Supported data types: bfloat16 or float8_e4m3fn.
            Shape: [sequence_length, token_dimension]
        scale (Tensor|None): Scaling factors required when hidden_states is of float8 type.
            For float8 inputs, this tensor provides the scaling factors for dequantization.
            Shape: [sequence_length, ceil(token_dimension / 128)]
            Data type: float32
        zipped_expertwise_rowmap (Tensor): Mapping tensor used to restore original order (unpermute).
            Shape: [sequence_length, num_experts]
            Data type: int32
        expert_id (int): The exact id of expert in the MoE layer, limited between 1 and 64.
        tokens_per_expert (list[int]): List where each element indicates the number of tokens
            assigned to the corresponding expert.
        padding_alignment (int): Tokens alignment requirement for expert buffers (in bytes).
            Must be a power of 2. Typical values are 16, 32 or 64 for optimal memory access.
        name (str|None, optional): Name prefix for the operation (optional).
            Default: None

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - hidden_states_unzipped (Tensor): The permuted and broadcasted input tensor.
                Shape: [total_tokens_after_broadcast, token_dimension]
                Data type: same as input hidden_states
            - scale_unzipped (Tensor): Broadcasted scale tensor (only valid for float8 inputs).
                Shape: [total_tokens_after_broadcast, ceil(token_dimension / 128)]
                Data type: float32
            - idx_unzipped(Tensor): Flattened expert indices aligned with given padding_alignment.
                Shape: [total_tokens_after_broadcast, 1]
                Data type: float32
    """
    if in_dynamic_or_pir_mode():
        (hidden_states_unzipped, scale_unzipped, idx_unzipped) = (
            _C_ops.tokens_unzip_gather(
                hidden_states,
                scale,
                zipped_expertwise_rowmap,
                expert_id,
                tokens_per_expert,
                padding_alignment,
            )
        )
        return (hidden_states_unzipped, scale_unzipped, idx_unzipped)
