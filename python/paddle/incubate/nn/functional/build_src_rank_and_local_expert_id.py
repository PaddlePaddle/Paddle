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

# from ....framework import LayerHelper, in_dynamic_or_pir_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor


def build_src_rank_and_local_expert_id(
    expert_num_global_tensor: Tensor,
    expert_num_global: list,
    num_local_experts: int,
    name: str | None = None,
) -> Tensor:
    """
    Args:
        expert_num_global_tensor:
        expert_num_global:
        num_local_experts:

    Returns:
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.build_src_rank_and_local_expert_id(
            expert_num_global_tensor, expert_num_global, num_local_experts
        )

    helper = LayerHelper('expert_num_global_tensor', **locals())
    vector = helper.create_variable_for_type_inference(dtype=paddle.int32)
    local_expert_id = helper.create_variable_for_type_inference(
        dtype=paddle.int32
    )

    inputs = {'expert_num_global_tensor': expert_num_global_tensor}
    attrs = {
        'expert_num_global': expert_num_global,
        'num_local_experts': num_local_experts,
    }
    outputs = {'vector': vector, 'local_expert_id': local_expert_id}
    helper.append_op(
        type='build_src_rank_and_local_expert_id',
        inputs=inputs,
        attrs=attrs,
        outputs=outputs,
    )
    return vector, local_expert_id
