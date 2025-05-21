# File: python/paddle/incubate/nn/functional/rms_norm_grad.py
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import convert_dtype
import paddle

def fused_rms_norm_grad(x, scale, invvar, dy, epsilon=1e-5, name=None):
    """
    Gradient for cuda_rms_norm.

    Args:
        x (Tensor): Original input tensor, shape [rows, cols].
        scale (Tensor): Scale tensor used in forward, shape [cols].
        invvar (Tensor): Inverse variance from forward, shape [rows].
        dy (Tensor): Upstream gradient of y, same shape as x.
        epsilon (float): Epsilon used in forward.

    Returns:
        dx (Tensor): Gradient w.r.t. x, same shape as x.
        dscale (Tensor): Gradient w.r.t. scale, same shape as scale.
    """
    helper = LayerHelper('fused_rms_norm_grad', **locals())
    # Create output tensors
    dx = helper.create_variable_for_type_inference(x.dtype)
    dscale = helper.create_variable_for_type_inference(scale.dtype)

    helper.append_op(
        type='fused_rms_norm_grad',
        inputs={
            'X': x,
            'Scale': scale,
            'InvVar': invvar,
            'DY': dy
        },
        outputs={
            'DX': dx,
            'DScale': dscale
        },
        attrs={'epsilon': epsilon}
    )
    return dx, dscale
