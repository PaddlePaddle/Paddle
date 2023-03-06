# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper


def cutlass_fused_glu(x, weight, bias, act_type):
    """
    Cutlass Fused GLU.

    This method requires SM_ARCH in sm70, sm75, sm80, and activation support `sigmoid`, `swish`, `gelu`.

    Args:
        x (Tensor): the Input Tensor. Its shape is [M, K].
        weight (Tensor): the Weight Tensor. Its shape is [K, 2N].
        bias (Tensor): the Bias Tensor. Its shape is [2N].
        act_type (Str): the activation type, support `sigmoid`, `swish`, `gelu`.
    Returns:
        Tensor: the output Tensor. Its shape is [M, N].

    Examples:
        .. code-block:: python
            # required: gpu

            import paddle
            import numpy as np
            from paddle.incubate.nn.functional import cutlass_fused_glu

            def naive_swiglu(x, weight, bias, act):
                out = paddle.matmul(x, weight)
                out = out + bias
                x0, x1 = paddle.chunk(out, 2, axis=1)
                x1 = paddle.nn.functional.swish(x1)
                return x0 * x1

            batch = 8
            hidden = 32
            dtype=paddle.float16

            x = paddle.randn([batch, hidden], dtype=dtype)
            weight = paddle.randn([hidden, hidden * 2], dtype=dtype)
            bias = paddle.randn([hidden * 2], dtype=dtype)

            naive_swiglu_out = naive_swiglu(x, weight, bias, "swish")
            # equals to: out = fused_swiglu_out = cutlass_fused_glu(x, weight, bias, "swish")

    """
    # requires_grad = not x.stop_gradient
    requires_grad = True
    if in_dygraph_mode():
        return _C_ops.cutlass_fused_glu(
            x, weight, bias, act_type, requires_grad
        )[0]

    helper = LayerHelper('cutlass_fused_glu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    matmul0_result = helper.create_variable_for_type_inference(dtype=x.dtype)
    matmul1_result = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='cutlass_fused_glu',
        inputs={'X': x, 'Weight': weight, 'Bias': bias},
        attrs={"Act_type": act_type, "Requires_grad": requires_grad},
        outputs={
            'Out': out,
            'matmul0_result': matmul0_result,
            'matmul1_result': matmul1_result,
        },
    )
    return out
