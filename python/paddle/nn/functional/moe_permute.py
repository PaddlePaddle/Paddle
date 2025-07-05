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


def moe_permute(
    hidden_states: Tensor,
    scale: Tensor | None,
    expert_routemap_topk: Tensor,
    expert_prob_topk: Tensor,
    num_experts: int,
    tokens_per_expert: list,
    padding_alignment: int,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Permute tokens for Mixture of Experts (MoE) computation.

    Args:
        hidden_states (Tensor): Input tensor storing tokens in row-major layout.
            Shape: [seq_len, token_len], dtype: bfloat16 or float8_e4m3fn.
        scale (Tensor|None): Input tensor required when hidden_states is fp8 type.
            Shape: [seq_len, (token_len + 127) // 128], dtype: float32.
        expert_routemap_topk (Tensor): Tensor recording which expert each token is dispatched to.
            Shape: [seq_len, topk], dtype: int32, value range: [-1, num_experts).
        expert_prob_topk (Tensor): Tensor storing expert probabilities.
            Shape: [seq_len, topk], dtype: float32.
        num_experts (int): Number of experts.
        tokens_per_expert (list[int]): List indicating how many tokens each expert receives.
        padding_alignment (int): Alignment requirement for expert buffers (must be multiple of this value).
        name (str|None, optional): Name for the operation. Defaults to None.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            - hidden_states_unzipped: Permuted and broadcasted tensor.
                Shape: [seqlen_broadcasted, token_len], dtype same as input.
            - zipped_expertwise_rowmap: Mapping tensor for unpermute operation.
                Shape: [seqlen, num_experts], dtype: int32.
            - token_prob_unzipped: Flattened expert probabilities aligned with hidden_states_unzipped.
                Shape: [seqlen_broadcasted, 1], dtype: float32.
            - scale_unzipped: Scaled tensor (only valid when hidden_states is fp8).
                Shape: [seqlen_broadcasted, (token_len + 127) // 128], dtype: float32.

    """
    if in_dynamic_or_pir_mode():
        (
            hidden_states_unzipped,
            zipped_expertwise_rowmap,
            token_prob_unzipped,
            scale_unzipped,
        ) = _C_ops.moe_permute(
            hidden_states,
            scale,
            expert_routemap_topk,
            expert_prob_topk,
            num_experts,
            tokens_per_expert,
            padding_alignment,
        )
        return (
            hidden_states_unzipped,
            zipped_expertwise_rowmap,
            token_prob_unzipped,
            scale_unzipped,
        )
