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


def moe_unpermute(
    hidden_states_unzipped: Tensor,
    zipped_expertwise_rowmap: Tensor,
    expert_routemap_topk: Tensor,
    token_prob_unzipped: Tensor,
    total_zipped_tokens: int,
    num_experts: int,
    use_mix_precision: bool = True,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Args:
        hidden_states_unzipped (Tensor): The input Tensor containing broadcasted and permuted hidden states.
            Shape: (seqlen_broadcasted, token_len). Dtype: bfloat16.
        zipped_expertwise_rowmap (Tensor): The input Tensor recording the mapping relationship for unpermute operation.
            Shape: (seqlen, num_experts). Dtype: int32.
        expert_routemap_topk (Tensor): The input Tensor indicating which expert each token is assigned to.
            Shape: (seqlen, 8). Value range: [-1, num_experts]. Dtype: int32.
        token_prob_unzipped (Tensor): The input Tensor containing flattened expert probabilities corresponding to hidden_states_unzipped.
            Shape: (seqlen_broadcasted, 1). Dtype: float32.
        total_zipped_tokens_num (int): The total number of tokens before permutation for output buffer allocation. Dtype: int32.
        num_experts (int): The number of experts. Dtype: int32.
        use_mix_precision (bool, optional): Whether to use mixed precision during accumulation.
            This option significantly improves precision when number of experts > 4. Default: True.
        name (str|None, optional): Name for the operation. Default: None.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - hidden_states (Tensor): The output Tensor with unpermuted tokens.
              Shape: (seqlen, token_len). Dtype: bfloat16.
            - expert_prob_topk (Tensor): The output Tensor with unpermuted probabilities.
              Shape: (seqlen, topk). Dtype: float32.

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
        zipped_tokens, zipped_probs_topk = _C_ops.moe_unpermute(
            hidden_states_unzipped,
            zipped_expertwise_rowmap,
            expert_routemap_topk,
            token_prob_unzipped,
            total_zipped_tokens,
            num_experts,
            use_mix_precision,
        )
        return (zipped_tokens, zipped_probs_topk)
