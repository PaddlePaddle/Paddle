from __future__ import annotations

from typing import TYPE_CHECKING, overload

import paddle
from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from paddle import Tensor

def fused_weighted_swiglu_act_quant(
    x: Tensor,
    prob: Tensor | None = None,
    using_pow2_scaling: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Fused weighted SwiGLU activation with quantization.

    This function performs a fused operation that combines weighted SwiGLU activation
    and quantization. The SwiGLU activation is applied to the input tensor x, 
    optionally weighted by prob, and then quantized to FP8 format.

    Args:
        x (Tensor): The input tensor. Must be 2D or higher dimensional with bfloat16 dtype.
                   The last dimension must be even (divisible by 2).
        prob (Tensor, optional): Optional probability tensor for weighting. If provided,
                               must be float32 dtype and have shape [batch_size]. Default: None.
        using_pow2_scaling (bool, optional): Whether to use power-of-2 scaling for quantization.
                                           Default: False.
        name (str, optional): Name for the operation (optional, default is None).
                            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - out (Tensor): The quantized output tensor with FP8_E4M3FN dtype.
                          Shape: [batch_size, last_dim // 2]
            - scale (Tensor): The quantization scale tensor with float32 dtype.
                            Shape: [batch_size, (last_dim // 2 + 127) // 128]

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> # Example 1: Basic usage without probability weighting
            >>> x = paddle.randn([32, 4096], dtype='bfloat16')
            >>> out, scale = F.fused_weighted_swiglu_act_quant(x)
            >>> print(f"Input shape: {x.shape}")
            >>> print(f"Output shape: {out.shape}")
            >>> print(f"Scale shape: {scale.shape}")
            Input shape: [32, 4096]
            Output shape: [32, 2048]
            Scale shape: [32, 16]

            >>> # Example 2: With probability weighting
            >>> x = paddle.randn([16, 2048], dtype='bfloat16')
            >>> prob = paddle.randn([16], dtype='float32')
            >>> out, scale = F.fused_weighted_swiglu_act_quant(x, prob=prob)
            >>> print(f"Output shape: {out.shape}")
            >>> print(f"Scale shape: {scale.shape}")
            Output shape: [16, 1024]
            Scale shape: [16, 8]

            >>> # Example 3: With power-of-2 scaling
            >>> x = paddle.randn([8, 1024], dtype='bfloat16')
            >>> out, scale = F.fused_weighted_swiglu_act_quant(
            ...     x, using_pow2_scaling=True
            ... )
            >>> print(f"Output shape: {out.shape}")
            Output shape: [8, 512]
    """

    def __check_input(x, prob, using_pow2_scaling):
        # Check input tensor x
        check_dtype(
            x.dtype,
            'x',
            ['bfloat16'],
            'fused_weighted_swiglu_act_quant',
        )

        input_shape = list(x.shape)
        assert len(input_shape) >= 2, (
            "The input tensor x must be at least 2-dimensional, "
            f"but received x's dimensional: {len(input_shape)}.\n"
        )

        last_dim = input_shape[-1]
        assert last_dim % 2 == 0, (
            "The last dimension of input tensor x must be even (divisible by 2), "
            f"but got last_dim = {last_dim}.\n"
        )

        # Check probability tensor if provided
        if prob is not None:
            check_dtype(
                prob.dtype,
                'prob',
                ['float32'],
                'fused_weighted_swiglu_act_quant',
            )
            
            prob_shape = list(prob.shape)
            batch_size = 1
            for i in range(len(input_shape) - 1):
                batch_size *= input_shape[i]
                
            assert len(prob_shape) == 1 and prob_shape[0] == batch_size, (
                f"The prob tensor must have shape [{batch_size}], "
                f"but got shape {prob_shape}.\n"
            )

        # Check using_pow2_scaling
        assert isinstance(using_pow2_scaling, bool), (
            "The using_pow2_scaling must be a boolean value, "
            f"but got {type(using_pow2_scaling)}.\n"
        )

    if in_dynamic_or_pir_mode():
        return _C_ops.fused_weighted_swiglu_act_quant(x, prob, using_pow2_scaling)
    else:
        __check_input(x, prob, using_pow2_scaling)

        helper = LayerHelper('fused_weighted_swiglu_act_quant', **locals())
        out = helper.create_variable_for_type_inference(dtype='float8_e4m3fn')
        scale = helper.create_variable_for_type_inference(dtype='float32')

        inputs = {'x': [x]}
        if prob is not None:
            inputs['prob'] = [prob]

        helper.append_op(
            type='fused_weighted_swiglu_act_quant',
            inputs=inputs,
            attrs={'using_pow2_scaling': using_pow2_scaling},
            outputs={'out': [out], 'scale': [scale]},
        )
        return out, scale