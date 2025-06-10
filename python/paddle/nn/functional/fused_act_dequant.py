import paddle
from paddle import _C_ops
from paddle.base.layer_helper import LayerHelper

__all__ = ['fused_act_dequant']


def fused_act_dequant(x, x_scale):
    """
    Fused activation and dequantization operation.
    
    This function performs dequantization on quantized float8_e4m3fn input tensor
    using the provided scales, converting it to bfloat16 format.
    
    Args:
        x (Tensor): Input quantized tensor with dtype float8_e4m3fn and shape [rows, cols].
        x_scale (Tensor): Scale tensor for dequantization with dtype float32.
            Can be 1D with shape [scale_groups] where scale_groups = (cols + 127) // 128,
            or 2D with shape [rows, scale_groups] for per-row scaling.
    
    Returns:
        Tensor: Dequantized output tensor with dtype bfloat16 and same shape as input x.
    
    Examples:
        >>> import paddle
        >>> import paddle.nn.functional as F
        
        >>> # Example 1: Basic usage with 1D scale
        >>> x = paddle.randint(0, 255, [512, 1024], dtype='uint8')  # Simulate quantized data
        >>> x = x.astype('float8_e4m3fn')  # Convert to float8_e4m3fn
        >>> x_scale = paddle.rand([8], dtype='float32')  # 1024 // 128 = 8 scale groups
        >>> out = F.fused_act_dequant(x, x_scale)
        >>> print(f"Input shape: {x.shape}, Output shape: {out.shape}")
        >>> print(f"Input dtype: {x.dtype}, Output dtype: {out.dtype}")
        
        >>> # Example 2: Per-row scaling with 2D scale
        >>> x = paddle.randint(0, 255, [256, 512], dtype='uint8')
        >>> x = x.astype('float8_e4m3fn')
        >>> x_scale = paddle.rand([256, 4], dtype='float32')  # 512 // 128 = 4 scale groups per row
        >>> out = F.fused_act_dequant(x, x_scale)
        >>> print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    
    Note:
        - Input x must be 2D tensor with dtype float8_e4m3fn
        - x_scale must have dtype float32
        - The number of columns in x should be divisible by 128 for optimal performance
        - Each scale value corresponds to 128 consecutive elements in the column dimension
    """
    # Input validation
    if not isinstance(x, paddle.Tensor):
        raise TypeError("x must be a Tensor")
    
    if not isinstance(x_scale, paddle.Tensor):
        raise TypeError("x_scale must be a Tensor")
    
    # Check data types
    if x.dtype != paddle.float8_e4m3fn:
        raise TypeError(f"x.dtype must be float8_e4m3fn, but got {x.dtype}")
    
    if x_scale.dtype != paddle.float32:
        raise TypeError(f"x_scale.dtype must be float32, but got {x_scale.dtype}")
    
    # Check dimensions
    if len(x.shape) != 2:
        raise ValueError(f"x must be 2D tensor, but got {len(x.shape)}D")
    
    if len(x_scale.shape) not in [1, 2]:
        raise ValueError(f"x_scale must be 1D or 2D tensor, but got {len(x_scale.shape)}D")
    
    # Get dimensions
    rows, cols = x.shape
    expected_scale_groups = (cols + 127) // 128
    
    # Validate x_scale shape
    if len(x_scale.shape) == 1:
        if x_scale.shape[0] != expected_scale_groups:
            raise ValueError(
                f"For 1D x_scale, size should be {expected_scale_groups} "
                f"(cols + 127) // 128, but got {x_scale.shape[0]}")
    else:  # 2D
        if x_scale.shape[0] != rows:
            raise ValueError(
                f"For 2D x_scale, first dimension should be {rows} "
                f"(same as x.shape[0]), but got {x_scale.shape[0]}")
        if x_scale.shape[1] != expected_scale_groups:
            raise ValueError(
                f"For 2D x_scale, second dimension should be {expected_scale_groups} "
                f"(cols + 127) // 128, but got {x_scale.shape[1]}")
    
    # Handle empty tensors
    if rows == 0 or cols == 0:
        return paddle.empty([rows, cols], dtype=paddle.bfloat16, place=x.place())
    
    # Call the kernel
    if paddle.in_dynamic_mode():
        return _C_ops.fused_act_dequant(x, x_scale)
    else:
        # Static graph mode
        helper = LayerHelper("fused_act_dequant", **locals())
        
        # Create output variable
        out = helper.create_variable_for_type_inference(dtype=paddle.bfloat16)
        
        # Append op
        helper.append_op(
            type="fused_act_dequant",
            inputs={"x": x, "x_scale": x_scale},
            outputs={"out": out},
            attrs={}
        )
        
        return out