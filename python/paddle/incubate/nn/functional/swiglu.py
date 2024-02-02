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

from ....framework import LayerHelper, in_dynamic_or_pir_mode


def swiglu(x, y=None, name=None):
    """
    This function performs SwiGLU activation to the input Tensor.

    .. math::

        out = silu(x) * y when y is not None
        out = silu(xs[0]) * xs[1] when y is None, where xs = paddle.chunk(2, axis=-1)

    Args:
        x (Tensor): The first input Tensor of SwiGLU.
        y (Tensor): The second input Tensor of SwiGLU.

    Returns:
        A Tensor with the same data type with x and y.
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.swiglu(x, y)
    else:
        helper = LayerHelper("swiglu", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="swiglu", inputs={"x": x, "y": y}, outputs={"out": out}
        )
        return out
