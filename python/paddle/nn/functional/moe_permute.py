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
    Permute tokens for Mixture of Experts (MoE) computation in distributed training scenarios.

    Note:
        This function reorganizes input tokens based on expert assignments to prepare for expert computation.
        It handles both bfloat16 and float8_e4m3fn data types with proper scaling for float8 inputs.

        1. This function is typically used in pair of moe_unpermute to provide complete MoE functionality.
        2. For float8 inputs, proper scaling must be provided via the scale parameter.
        3. The padding_alignment parameter affects memory efficiency but not correctness.
        4. Any output tokens can find an exact-match in the original input tokens.
        5. This permute function has overcomed the aadiff issue, is deterministic.

    Args:
        hidden_states (Tensor): The input tensor containing tokens to be permuted, stored in row-major layout.
            Supported data types: bfloat16 or float8_e4m3fn.
            Shape: [sequence_length, token_dimension]
        scale (Tensor|None): Scaling factors required when hidden_states is of float8 type.
            For float8 inputs, this tensor provides the scaling factors for dequantization.
            Shape: [sequence_length, ceil(token_dimension / 128)]
            Data type: float32
        expert_routemap_topk (Tensor): Tensor indicating expert assignments for each token (top-k experts).
            Each value represents the expert index the token is assigned to (-1 indicates not assigned).
            Shape: [sequence_length, top_k_experts]
            Data type: int32
            Value range: [-1, num_experts)
        expert_prob_topk (Tensor): Tensor containing routing probabilities for top-k experts.
            Shape: [sequence_length, top_k_experts]
            Data type: float32
        num_experts (int): Total number of experts in the MoE layer, limited between 1 and 64.
        tokens_per_expert (list[int]): List where each element indicates the number of tokens
            assigned to the corresponding expert.
        padding_alignment (int): Tokens alignment requirement for expert buffers (in bytes).
            Must be a power of 2. Typical values are 16, 32 or 64 for optimal memory access.
        name (str|None, optional): Name prefix for the operation (optional).
            Default: None

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            - hidden_states_unzipped (Tensor): The permuted and broadcasted input tensor.
                Shape: [total_tokens_after_broadcast, token_dimension]
                Data type: same as input hidden_states
            - zipped_expertwise_rowmap (Tensor): Mapping tensor used to restore original order (unpermute).
                Shape: [sequence_length, num_experts]
                Data type: int32
            - token_prob_unzipped (Tensor): Flattened expert probabilities aligned with permuted tokens.
                Shape: [total_tokens_after_broadcast, 1]
                Data type: float32
            - scale_unzipped (Tensor): Broadcasted scale tensor (only valid for float8 inputs).
                Shape: [total_tokens_after_broadcast, ceil(token_dimension / 128)]
                Data type: float32

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> # doctest: +SKIP('This is only support in cuda 12.0+')
            >>> import paddle
            >>> import numpy as np
            >>> import paddle.nn.functional as F
            >>> hidden_states = paddle.randn([3, 128], dtype='bfloat16')
            >>> expert_routemap_topk = paddle.to_tensor([[-1, 0, -1, -1, 2, -1, -1, -1],
            ...                                          [1, -1, -1, -1, -1, -1, -1, -1],
            ...                                          [-1, -1, -1, -1, -1, -1, 1, -1]],
            ...                                           dtype='int32')
            >>> expert_prob_topk= paddle.to_tensor([[0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
            ...                                     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ...                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            ...                                          dtype='float32')
            >>> num_experts = 3
            >>> tokens_per_expert = [1, 2, 1]
            >>> padding_alignment = 2
            >>> hidden_states_unzipped, zipped_expertwise_rowmap, token_prob_unzipped, scale_unzipped = F.moe_permute(
            ...     hidden_states,
            ...     None,
            ...     expert_routemap_topk,
            ...     expert_prob_topk,
            ...     num_experts,
            ...     tokens_per_expert,
            ...     padding_alignment,
            ... )
            >>> # weighted by probs.
            >>> hidden_states_unzipped = (hidden_states_unzipped.astype("float32") * token_prob_unzipped.astype("float32").unsqueeze(-1)).astype("bfloat16")
            >>> zipped_tokens, zipped_probs = F.moe_unpermute(hidden_states_unzipped, zipped_expertwise_rowmap, expert_routemap_topk, token_prob_unzipped,3,3)
            >>> np.testing.assert_allclose(zipped_tokens.numpy(), hidden_states.numpy(), rtol=1e-05, atol=1e-06)
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
