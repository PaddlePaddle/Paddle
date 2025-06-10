import paddle
from paddle import _C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.base.framework import in_dygraph_mode

__all__ = ['fused_transpose_split_quant']


def fused_transpose_split_quant(x, tokens_per_expert, pow_2_scales=False):
    
    if not isinstance(x, paddle.Tensor):
        raise TypeError("x must be a Tensor")
    
    if x.dtype != paddle.bfloat16:
        raise TypeError(f"x.dtype must be bfloat16, but got {x.dtype}")
    
    if len(x.shape) != 2:
        raise ValueError(f"x must be 2D tensor, but got {len(x.shape)}D")
    
    if not isinstance(tokens_per_expert, (list, tuple)):
        raise TypeError("tokens_per_expert must be a list or tuple")
    
    if len(tokens_per_expert) == 0:
        raise ValueError("tokens_per_expert cannot be empty")
    
    tokens_per_expert = [int(t) for t in tokens_per_expert]
    
    for i, tokens in enumerate(tokens_per_expert):
        if tokens <= 0:
            raise ValueError(f"tokens_per_expert[{i}] must be positive, but got {tokens}")
        if tokens % 128 != 0:
            raise ValueError(f"tokens_per_expert[{i}] must be divisible by 128, but got {tokens}")
    
    # 验证总token数
    total_tokens = sum(tokens_per_expert)
    if total_tokens != x.shape[0]:
        raise ValueError(f"sum(tokens_per_expert) ({total_tokens}) must equal x.shape[0] ({x.shape[0]})")
    
    # 验证K的大小
    K = x.shape[1]
    if K > 65535 * 128:
        raise ValueError(f"x.shape[1] ({K}) must be <= {65535 * 128}")
    
    if not isinstance(pow_2_scales, bool):
        raise TypeError("pow_2_scales must be a bool")
    
    # 处理空输入的情况
    if x.shape[0] == 0 or x.shape[1] == 0:
        return [], []
    
    if in_dygraph_mode():
        return _C_ops.fused_transpose_split_quant(
            x, tokens_per_expert, pow_2_scales
        )
    else:

        helper = LayerHelper("fused_transpose_split_quant", **locals())
        
        outs = []
        scales = []
        
        for i, tokens in enumerate(tokens_per_expert):
            # outs[i]: [K, tokens]
            out = helper.create_variable_for_type_inference(
                dtype=paddle.float8_e4m3fn
            )
            outs.append(out)
            
            # scales[i]: [tokens//128, K]
            scale = helper.create_variable_for_type_inference(
                dtype=paddle.float32
            )
            scales.append(scale)
        
        helper.append_op(
            type="fused_transpose_split_quant",
            inputs={"x": x},
            outputs={"outs": outs, "scales": scales},
            attrs={
                "tokens_per_expert": tokens_per_expert,
                "pow_2_scales": pow_2_scales
            }
        )
        
        return outs, scales