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
        scatter_index: Scatter indices [k, s]
    
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

if __name__ == "__main__":
    print("This module is not for direct use.")
    x = paddle.arange(1, 16).view((5, 3)).astype('float32')
    combine_weights = paddle.to_tensor([
        [0, 0],
        [0, 0],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])

    # 分散索引
    scatter_index = paddle.to_tensor([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0]
    ]).astype('int32')

    # 输出计算
    output = paddle.zeros((5, 3))
    for s in range(5):
        expert0_idx = scatter_index[0, s]
        expert1_idx = scatter_index[1, s]
        output[s] = (
            x[expert0_idx] * combine_weights[s, 0] + 
            x[expert1_idx] * combine_weights[s, 1]
        )
    print(output)
    print(moe_combine(x, combine_weights, scatter_index))