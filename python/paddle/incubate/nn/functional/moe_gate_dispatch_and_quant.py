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
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.incubate.nn.functional import (
    fp8,
    moe_gate_dispatch,
)

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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Args:
        x: Input tensor [batch_size, seq_len, hidden_dim].
        gate_logits: Gate logits for choosing experts [batch_size, seq_len, num_experts].
        corr_bias: Optional correction bias to adjust gate logits.
        k: Top-k experts to be selected.
        capacity: The maximum number of tokens an expert can handle.
        use_pad: Boolean indicating if padding is used for uniform input length.
        use_pow2_scale: Boolean indicating if power-of-two scaling is applied for quantization.

    Returns:
        Tuple of Tensors containing:
        - fp8_out: Processed output tensor in FP8 format.
        - scale: Scaling factors used during processing.
        - combine_weights: Weights for combining experts' outputs.
        - scatter_index: Indices for scattering inputs to experts.
        - expert_offset: Offset indices for each expert.
        - expert_id: IDs of selected experts for each position.
    """
    if not in_dynamic_or_pir_mode():
        raise NotImplementedError('Static graph mode not implemented')
    else:
        if paddle.device.is_compiled_with_custom_device('npu'):
            return math_moe_gate_dispatch_and_quant(
                x, gate_logits, corr_bias, k, capacity, use_pad
            )
        else:
            return _C_ops.moe_gate_dispatch_and_quant(
                x, gate_logits, corr_bias, k, capacity, use_pad, use_pow2_scale
            )


def math_moe_gate_dispatch_and_quant(
    x, gate_logits, corr_bias, k, capacity, use_pad, use_pow2_scale
):
    y, combine_weights, scatter_index, expert_offset, expert_id = (
        moe_gate_dispatch(x, gate_logits, corr_bias, k, capacity, use_pad)
    )
    y_fp8, scale = fp8.fp8_quant_blockwise(
        y,
        quant_method="1x128",
        output_scale_transpose=False,
        using_pow2_scale=use_pow2_scale,
    )
    return (
        y_fp8,
        scale,
        combine_weights,
        scatter_index,
        expert_offset,
        expert_id,
    )
