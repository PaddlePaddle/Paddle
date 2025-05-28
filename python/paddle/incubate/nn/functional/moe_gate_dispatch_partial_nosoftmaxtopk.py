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


def moe_gate_dispatch_partial_nosoftmaxtopk(
    x: Tensor,
    combine_weights: Tensor,
    expert_id: Tensor,
    k: int,
    capacity: int,
    num_experts: int,
    use_pad: bool,
    expert_start_index: int,
    expert_end_index: int,
    reverse_token_drop: bool,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if in_dynamic_or_pir_mode():
        return _C_ops.moe_gate_dispatch_partial_nosoftmaxtopk(
            x,
            combine_weights,
            expert_id,
            k,
            capacity,
            num_experts,
            use_pad,
            expert_start_index,
            expert_end_index,
            reverse_token_drop,
        )
    helper = LayerHelper("moe_gate_dispatch_partial_nosoftmaxtopk", **locals())
    y = helper.create_variable_for_type_inference(dtype=x.dtype)
    combine_weights_out = helper.create_variable_for_type_inference(
        dtype=combine_weights.dtype
    )
    scatter_index = helper.create_variable_for_type_inference(dtype='int32')
    scatter_index_rev = helper.create_variable_for_type_inference(dtype='int32')
    expert_offset = helper.create_variable_for_type_inference(dtype='int64')
    expert_nums_local = helper.create_variable_for_type_inference(dtype='int64')
    inputs = {
        "x": x,
        "combine_weights": combine_weights,
        "expert_id": expert_id,
    }
    outputs = {
        "y": y,
        "combine_weights_out": combine_weights_out,
        "scatter_index": scatter_index,
        "scatter_index_rev": scatter_index_rev,
        "expert_offset": expert_offset,
        "expert_nums_local": expert_nums_local,
    }
    attrs = {
        "k": k,
        "capacity": capacity,
        "num_experts": num_experts,
        "use_pad": use_pad,
        "expert_start_index": expert_start_index,
        "expert_end_index": expert_end_index,
        "reverse_token_drop": reverse_token_drop,
    }
    helper.append_op(
        type="moe_gate_dispatch_partial_nosoftmaxtopk",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs,
    )
    return (
        y,
        combine_weights_out,
        scatter_index,
        scatter_index_rev,
        expert_offset,
        expert_nums_local,
    )


# import paddle
# import numpy as np

# num_rows = 4
# feature_dim = 8
# num_experts = 3
# k = 2
# capacity = 5

# # 输入张量
# x = paddle.to_tensor(np.random.rand(num_rows, feature_dim).astype('float32'), stop_gradient=False)

# # 合并权重张量
# combine_weights = paddle.to_tensor(np.random.rand(num_rows, k).astype('float32'), stop_gradient=False)

# # 专家ID张量
# expert_id = paddle.to_tensor(np.random.randint(0, num_experts, size=(num_rows, k)).astype('int32'), stop_gradient=False)

# print("x type:", x.dtype)
# print("combine_weights type:", combine_weights.dtype)
# print("expert_id type:", expert_id.dtype)
# # 其他参数
# use_pad = True
# expert_start_index = 0
# expert_end_index = num_experts
# reverse_token_drop = False

# # 调用自定义算子
# y, combine_weights_out, scatter_index, scatter_index_rev, expert_offset, expert_nums_local = moe_ops_partial_nosoftmaxtopk(
#     x=x,
#     combine_weights=combine_weights,
#     expert_id=expert_id,
#     k=k,
#     capacity=capacity,
#     num_experts=num_experts,
#     use_pad=use_pad,
#     expert_start_index=expert_start_index,
#     expert_end_index=expert_end_index,
#     reverse_token_drop=reverse_token_drop
# )

# # 打印结果
# print("y:", y.numpy())
# print("combine_weights_out:", combine_weights_out.numpy())
# print("scatter_index:", scatter_index.numpy())
# print("scatter_index_rev:", scatter_index_rev.numpy())
# print("expert_offset:", expert_offset.numpy())
# print("expert_nums_local:", expert_nums_local.numpy())

# a = paddle.sum(y)+paddle.sum(combine_weights_out)
# a.backward()
# print("\n##########backward output##########\n")
# print(f"x.grad: {x.grad}\n combine_weights.grad: {combine_weights.grad}\n expert_id.grad: {expert_id.grad}")
