# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from .tensor.compat_softmax import log_softmax, softmax
from .tensor.creation import assign
from .tensor.math import (
    erf,
    expm1,
    i0,
    i0e,
    i1,
    i1e,
    log1p,
    logit,
    logsumexp,
    sinc as _sinc,
)
from .utils.decorator_utils import param_one_alias

__all__ = [
    "erf",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "log1p",
    "log_softmax",
    "logit",
    "logsumexp",
    "sinc",
    "softmax",
    "expm1",
]


@param_one_alias(["x", "input"])
def sinc(x, name=None, *, out=None):
    r"""
    Calculate the normalized sinc of ``x`` elementwise.

    .. math::

        out_i =
        \left\{
        \begin{aligned}
        &1 & \text{ if $x_i = 0$} \\
        &\frac{\sin(\pi x_i)}{\pi x_i} & \text{ otherwise}
        \end{aligned}
        \right.

    Args:
        x (Tensor): The input Tensor. Must be one of the following types: bfloat16, float16, float32, float64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|None, optional): The output Tensor. If set, the result will be stored in this Tensor. Default is None.

    Returns:
        out (Tensor), The Tensor of elementwise-computed normalized sinc result.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([0.0, 0.5, 1.0], dtype='float32')
            >>> paddle.special.sinc(x)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.        , 0.63661975, 0.        ])
    """
    result = _sinc(x, name=name)
    return assign(result, out) if out is not None else result
