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
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor


def fused_moe_permute(
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
    # 为了突出重点，省略部分代码
    # 动静统一分支，直接调用算子对应的 Python C 函数
    if in_dynamic_or_pir_mode():
        X_unzipped, zipped_experwise, token_prob_unzipped, XScale_unzipped = (
            _C_ops.fused_moe_permute(
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

    # 老静态图分支
    ## 输入参数检查
    # __check_input

    ## 构造输出，添加 op，返回输出
    helper = LayerHelper('fused_moe_permute', **locals())
    X_unzipped = helper.create_variable_for_type_inference(dtype=X.dtype)
    zipped_experwise = helper.create_variable_for_type_inference(
        dtype=expert_routemap_topk.dtype
    )
    token_prob_unzipped = helper.create_variable_for_type_inference(
        dtype=expert_prob_topk.dtype
    )
    XScale_unzipped = helper.create_variable_for_type_inference(
        dtype=XScale.dtype
    )

    inputs = {
        'X': X,
        'XScale': XScale,
        'expert_routemap_topk': expert_routemap_topk,
        'expert_prob_topk': expert_prob_topk,
    }

    outputs = {
        'X_unzipped': X_unzipped,
        'zipped_experwise': zipped_experwise,
        'token_prob_unzipped': token_prob_unzipped,
        'XScale_unzipped': XScale_unzipped,
    }
    attrs = {
        'topk': topk,
        'num_experts': num_experts,
        'tokens_per_expert': tokens_per_expert,
        'padding_multiplex': padding_multiplex,
    }
    helper.append_op(
        type='fused_moe_permute',
        inputs=inputs,
        attrs=attrs,
        outputs=outputs,
    )
    return (X_unzipped, zipped_experwise, token_prob_unzipped, XScale_unzipped)
