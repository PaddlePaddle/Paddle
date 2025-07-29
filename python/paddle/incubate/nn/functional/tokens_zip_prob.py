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
    from collections.abc import Sequence

    from paddle import Tensor


def tokens_zip_prob(
    unzipped_prob: Sequence[Tensor],
    zipped_expertwise_rowmap: Tensor,
    dispatched_indices: Tensor,
    name: str | None = None,
) -> tuple[Tensor]:
    r"""
    Partially unpermute probs for Mixture of Experts (MoE) computation in distributed training scenarios.

    Args:
        unzipped_prob (list[Tensor] or tuple[Tensor]): Flattened expert probabilities aligned with permuted tokens.
            Shape: [total_tokens_after_broadcast, 1]
            Data type: float32
        zipped_expertwise_rowmap (Tensor): Mapping tensor used to restore original order (unpermute).
            Shape: [sequence_length, num_experts]
            Data type: int32
        dispatched_indices (Tensor): Tensor indicating expert assignments for each token (top-k experts).
            Each value represents the expert index the token is assigned to (-1 indicates not assigned).
            Shape: [sequence_length, top_k_experts]
            Data type: int32
            Value range: [-1, num_experts)
        name (str|None, optional): Name prefix for the operation (optional).
            Default: None
    Returns:
        tuple[Tensor]:
            - zipped_prob (Tensor): The output Tensor with unpermuted probabilities.

    """
    if in_dynamic_or_pir_mode():
        (zipped_prob,) = _C_ops.tokens_zip_prob(
            unzipped_prob, zipped_expertwise_rowmap, dispatched_indices
        )
        return zipped_prob
