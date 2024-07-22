# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import _C_ops
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.layer_helper import LayerHelper, in_dygraph_mode


def lsqplus(x, alpha, beta, g_scale, bit_length, is_sign, round_type):
    """
    ***lsqplus***

    This is a Lsqplus quantization algorithm operator, which is implemented on kernel level.
    This operator supports both forward and backward computation.

    :math: qx = round(clip((x - beta) / alpha, qmin, qmax)) * scale + b

    Args:
        x (Tensor): The input tensor of Lsqplus quantization operator.
        alpha (float): The stepsize parameter.
        beta (float): The bias parameter.
        g (float): The gradient scaling parameter.
        bit_length (int): The bit length of the quantization.
        is_sign (bool): if True, use sign mode(e.g. [-128, 127] for int8).
        round_type (str): The round type of the quantization
        (0 for round then clip with TiesToEven, otherwise clip then round with TiesAwayFromZero).

    Returns:
        output (Tensor): The fakequantized tensor of Lsqplus quantization operator.
    """

    def __check_input(x, alpha, beta, g_scale, bit_length, is_sign, round_type):
        # check dtype
        check_variable_and_dtype(
            x,
            'x',
            ['float32', 'float16'],
            'lsqplus',
            extra_message="The dtype of the input must a float16, float32",
        )
        check_variable_and_dtype(
            alpha,
            'alpha',
            ['float32', 'float16'],
            'lsqplus',
            extra_message="The dtype of the alpha must a float16, float32",
        )
        check_variable_and_dtype(
            beta,
            'beta',
            ['float32', 'float16'],
            'lsqplus',
            extra_message="The dtype of the beta must a float16, float32",
        )
        check_variable_and_dtype(
            g_scale,
            'g_scale',
            ['float32', 'float16'],
            'lsqplus',
            extra_message="The dtype of the g must a float16, float32",
        )

        # check shape
        assert (
            len(x.shape) >= 1 and x.numel() >= 1
        ), "the input's shape length should not be zero"
        assert (
            len(alpha.shape) == 1 and alpha.numel() == 1
        ), "the alpha should be a scalar"
        assert (
            len(beta.shape) == 1 and beta.numel() == 1
        ), "the beta should be a scalar"
        assert (
            len(g_scale.shape) == 1 and g_scale.numel() == 1
        ), "the g should be a scalar"

        # check number
        assert (
            bit_length >= 2 and bit_length <= 8
        ), "the bit_length should be in [2, 8]"

    __check_input(x, alpha, beta, g_scale, bit_length, is_sign, round_type)

    # dynamic branch
    if in_dygraph_mode():
        return _C_ops.fake_quantize_dequantize_lsqplus(
            x, alpha, beta, g_scale, bit_length, is_sign, round_type
        )

    # static  branch
    helper = LayerHelper('fake_quantize_dequantize_lsqplus', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fake_quantize_dequantize_lsqplus',
        inputs={
            'x': [x],
            'alpha': [alpha],
            'beta': [beta],
            'g_scale': [g_scale],
        },
        attrs={
            'bit_length': bit_length,
            'is_sign': is_sign,
            'round_type': round_type,
        },
        outputs={'out': [out]},
    )
    return out
