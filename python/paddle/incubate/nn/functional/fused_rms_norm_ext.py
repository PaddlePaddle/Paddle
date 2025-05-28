# File: python/paddle/incubate/nn/functional/layer_norm_cuda.py
import paddle
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle import _C_ops

def fused_rms_norm_ext(x, scale, epsilon=1e-5, name=None):
    """
    Applies Layer Normalization over the last dimension of the input tensor using CUDA implementation.
    Args:
        x (Tensor): Input tensor of shape [rows, cols] or higher dimensions (flattened to 2D).
        scale (Tensor): Scale tensor of shape [cols].
        bias (Tensor, optional): Bias tensor of shape [cols]. If None, no bias is added.
        epsilon (float): Small constant to avoid division by zero.
        name (str, optional): Name of the operator.
    Returns:
        y (Tensor): Normalized tensor of same shape as x.
        mean (Tensor): Tensor of shape [rows], the mean of each row.
        invvar (Tensor): Tensor of shape [rows], the inverse standard deviation of each row.
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.fused_rms_norm(
            x,scale,epsilon
        )
    helper = LayerHelper('fused_rms_norm', **locals())
    dtype = convert_dtype(x.dtype)
    y = helper.create_variable_for_type_inference(dtype)
    invvar = helper.create_variable_for_type_inference('float32')

    inputs = {'x': x, 'scale': scale}

    helper.append_op(
        type='fused_rms_norm',
        inputs=inputs,
        outputs={'y': y, 'invvar': invvar},
        attrs={'epsilon': epsilon}
    )
    return y, invvar