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
import paddle.distributed as dist

# from ....framework import LayerHelper, in_dynamic_or_pir_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor


def cal_aux_loss(
    gate_prob: Tensor,
    dispatch_mask: Tensor,
    tokens_mask: Tensor,
    dispatch_tokens_mask: Tensor,
    num_experts: int,
    use_group: bool,
    moe_k: int,
    clip_min: float,
    name: str | None = None,
) -> Tensor:
    """
    Args:
        gate_prob:
        dispatch_mask:
        tokens_mask:
        dispatch_tokens_mask:
        num_experts:
        use_group:
        moe_k:
        clip_min:

    Returns:
    """
    if in_dynamic_or_pir_mode():
        if paddle.device.is_compiled_with_custom_device('npu'):
            return math_cal_aux_loss(gate_prob, dispatch_mask, tokens_mask, dispatch_tokens_mask, num_experts,
                                     use_group, moe_k)
        else:
            return _C_ops.cal_aux_loss(
            gate_prob,
            dispatch_mask,
            tokens_mask,
            dispatch_tokens_mask,
            num_experts,
            use_group,
            moe_k,
            clip_min,
        )

    helper = LayerHelper('cal_aux_loss', **locals())
    l_aux_loss = helper.create_variable_for_type_inference(
        dtype=gate_prob.dtype
    )
    seqlen_float = helper.create_variable_for_type_inference(
        dtype=gate_prob.dtype
    )
    ce = helper.create_variable_for_type_inference(dtype=gate_prob.dtype)

    inputs = {
        'gate_prob': gate_prob,
        'dispatch_mask': dispatch_mask,
        'tokens_mask': tokens_mask,
        'dispatch_tokens_mask': dispatch_tokens_mask,
    }
    attrs = {
        'num_experts': num_experts,
        'use_group': use_group,
        'moe_k': moe_k,
        'clip_min': clip_min,
    }
    outputs = {'l_aux_loss': l_aux_loss, 'seqlen_float': seqlen_float, 'ce': ce}
    helper.append_op(
        type='cal_aux_loss', inputs=inputs, attrs=attrs, outputs=outputs
    )
    return l_aux_loss, seqlen_float, ce


def math_cal_aux_loss(
    gate_prob,
    dispatch_mask,
    tokens_mask,
    dispatch_tokens_mask,
    num_experts,
    use_group,
    moe_k,
    global_aux_loss=False,
    rank=None,
    group=None,
    clip_min=1e-6,
):
    if tokens_mask is not None and tokens_mask.dtype != gate_prob.dtype:
        tokens_mask = tokens_mask.astype(gate_prob.dtype)

    scale = None
    if dispatch_tokens_mask is not None:
        seqlen_float = dispatch_tokens_mask.astype(gate_prob.dtype).sum()
        if (
            tokens_mask is not None
            and gate_prob.shape[0] != dispatch_tokens_mask.shape[0]
        ):
            scale = seqlen_float / paddle.clip(tokens_mask.sum(), min=1e-6)
    elif tokens_mask is not None:
        seqlen_float = tokens_mask.sum()
    else:
        seqlen_float = gate_prob.numel().astype(gate_prob.dtype) / num_experts
    seqlen_float = paddle.clip(seqlen_float, min=1e-6)
    if len(dispatch_mask.shape) == 2:
        dispatch_mask = dispatch_mask.sum(0)
    ce = dispatch_mask.astype(gate_prob.dtype).detach() / seqlen_float
    me = paddle.sum(gate_prob, axis=0) / seqlen_float
    if global_aux_loss:
        me_list, ce_list = [], []
        dist.all_gather(me_list, me, group=group)
        dist.all_gather(ce_list, ce, group=group)
        me_list[rank] = me
        ce_list[rank] = ce
        me = paddle.stack(me_list).mean(0)
        ce = paddle.stack(ce_list).mean(0)

    l_aux = paddle.sum(me * ce) * num_experts
    if use_group:
        l_aux = l_aux / moe_k
    if scale is not None:
        l_aux = l_aux + (scale - 1) * l_aux.detach()
    return l_aux, seqlen_float, ce
