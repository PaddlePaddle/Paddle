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
    unzipped_tokens: Tensor,
    zipped_expertwise_rowmap: Tensor,
    expert_routemap_topk: Tensor,
    unzipped_token_probs: Tensor,
    total_zipped_tokens: int,
    num_experts: int,
    MP: bool = True,
    name: str | None = None,
):
    if in_dynamic_or_pir_mode():
        zipped_tokens, zipped_probs_topk = _C_ops.moe_unpermute(
            unzipped_tokens,
            zipped_expertwise_rowmap,
            expert_routemap_topk,
            unzipped_token_probs,
            total_zipped_tokens,
            num_experts,
            MP,
        )
        return (zipped_tokens, zipped_probs_topk)
