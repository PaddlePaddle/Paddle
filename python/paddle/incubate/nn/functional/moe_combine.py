from __future__ import annotations
from typing import TYPE_CHECKING
from paddle import _C_ops
import paddle
# from ....framework import LayerHelper, in_dynamic_or_pir_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor

def moe_combine(
    x: Tensor, combine_weights: Tensor, scatter_index: Tensor, name: str | None = None
) -> Tensor:
    """
    Args:
        x: Input tensor [seq, dim]
        combine_weights: Combination weights [s, k]
        scatter_index: Scatter indices [k, s] dtype=int32
    
    Returns:
        Output Combined output [s, dim]
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.moe_combine(x, combine_weights, scatter_index)
    helper = LayerHelper('moe_combine', **locals())
    y = helper.create_variable_for_type_inference(dtype=x.dtype)
    inputs = {
        'x': x,
        'combine_weights': combine_weights,
        'scatter_index': scatter_index
    }
    helper.append_op(type='moe_combine', inputs=inputs, outputs={'y': y})
    return y