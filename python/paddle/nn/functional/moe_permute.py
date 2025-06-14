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
    X: Tensor,
    XScale: Tensor | None,
    expert_routemap_topk: Tensor,
    expert_prob_topk: Tensor,
    topk: int,
    num_experts: int,
    tokens_per_expert: list,
    padding_multiplex: int,
    name: str | None = None,
):
    if in_dynamic_or_pir_mode():
        X_unzipped, zipped_experwise, token_prob_unzipped, XScale_unzipped = (
            _C_ops.moe_permute(
                X,
                XScale,
                expert_routemap_topk,
                expert_prob_topk,
                topk,
                num_experts,
                tokens_per_expert,
                padding_multiplex,
            )
        )
        return (
            X_unzipped,
            zipped_experwise,
            token_prob_unzipped,
            XScale_unzipped,
        )
