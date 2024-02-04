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
        out = silu(xs[0]) * xs[1] when y is None, where xs = paddle.chunk(x, 2, axis=-1)

    Args:
        x (Tensor): The first input Tensor of SwiGLU.
        y (Tensor, optional): The second input Tensor of SwiGLU. Default: None.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type with x and y.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.incubate.nn.functional as F
            >>> x = paddle.to_tensor([1, 2], dtype='float32')
            >>> out1, out2 = F.swiglu(x), F.swiglu(x, x)
            >>> print(out1, out2)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [1.46211720]) Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [0.73105860, 3.52318811])
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
