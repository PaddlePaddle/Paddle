from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import paddle
from paddle import _C_ops
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor

def moe_ops_partial_nosoftmaxtopk(
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
    name: str | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if in_dynamic_or_pir_mode():
        return _C_ops.moe_gate_dispatch_partial_nosoftmaxtopk(x, combine_weights, expert_id, k, capacity, num_experts, use_pad, expert_start_index, expert_end_index, reverse_token_drop)
    helper = LayerHelper("moe_ops_partial_nosoftmaxtopk", **locals())
    y = helper.create_variable_for_type_inference(dtype=x.dtype)
    combine_weights_out = helper.create_variable_for_type_inference(dtype=combine_weights.dtype)
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
        type="moe_ops_partial_nosoftmaxtopk",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs,
    )
    return (y, combine_weights_out, scatter_index, scatter_index_rev, expert_offset, expert_nums_local)

   