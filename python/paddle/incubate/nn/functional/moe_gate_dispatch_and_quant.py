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


def moe_gate_dispatch_and_quant(
    x: Tensor,
    gate_logits: Tensor,
    corr_bias: Tensor,
    k: int,
    capacity: int,
    use_pad: bool,
    use_pow2_scale: bool,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Args:
        x:
            Input tensor, usually of shape [batch_size, sequence_length, feature_dim].
        gate_logits:
            Logits for gating mechanism, determining input routing to experts.
        corr_bias:
            Bias for adjusting gate logits.
        k:
            Number of top experts to select for each input.
        capacity:
            Max tokens each expert can process per batch.
        use_pad:
            Boolean indicating if padding is used for uniform input length.
        use_pow2_scale:
            Boolean indicating if power-of-two scaling is applied for quantization.

    Returns:
        fp8_out:
            Processed output tensor in FP8 format.
        scale:
            Scaling factors used during processing.
        combine_weights:
            Weights for combining expert outputs.
        scatter_index:
            Indices for scattering outputs back to original order.
        expert_offset:
            Start index of each expert's output.
        expert_id:
            IDs of selected experts for each input.
    """
    if not in_dynamic_or_pir_mode():
        raise NotImplementedError('Static graph mode not implemented')
    else:
        return _C_ops.moe_gate_dispatch_and_quant(
            x, gate_logits, corr_bias, k, capacity, use_pad, use_pow2_scale
        )
