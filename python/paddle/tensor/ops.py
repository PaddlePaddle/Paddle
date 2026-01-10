#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

from typing import TYPE_CHECKING

from paddle._C_ops import (  # noqa: F401
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    ceil,
    cos,
    cosh,
    exp,
    expm1,
    floor,
    reciprocal,
    round,
    rsqrt,
    sigmoid,
    sin,
    sinh,
    sqrt,
    square,
    tan,
)
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from .. import _C_ops
from ..framework import in_dynamic_or_pir_mode
from .layer_function_generator import (
    generate_inplace_fn,
    generate_layer_fn,
)

if TYPE_CHECKING:
    from paddle import Tensor


__inplace_unary_func__ = [
    'exp_',
    'sqrt_',
    'rsqrt_',
    'ceil_',
    'floor_',
    'reciprocal_',
    'sigmoid_',
    'abs_',
    'sin_',
    'sinh_',
    'asin_',
    'asinh_',
    'cos_',
    'cosh_',
    'acos_',
    'acosh_',
    'tan_',
    'atan_',
    'atanh_',
    'expm1_',
    'erf_',
    'square_',
]

__all__ = []

# It is a hot fix in some unittest using:
#   paddle.scale(x=x, scale=10.0, out=out_var)
# e.g.: test_program_code.py, test_dist_train.py
globals()['_scale'] = generate_layer_fn('scale')

for _OP in set(__inplace_unary_func__):
    func = generate_inplace_fn(_OP)
    func.__module__ = __name__
    _func = inplace_apis_in_dygraph_only(func)
    globals()[_OP] = _func


@inplace_apis_in_dygraph_only
def round_(x, decimals=0, name=None):
    r"""
    Inplace version of ``round`` API, output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_round`.
    """
    return _C_ops.round_(x, decimals)


def erf(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    The error function.
    For more details, see `Error function <https://en.wikipedia.org/wiki/Error_function>`_.

    Equation:
        ..  math::
            out = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64, uint8, int8, int16, int32, int64.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. The output of Erf, dtype: float32 or float64 (integer types are autocasted into float32), shape:  same as  input.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.erf(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.42839241, -0.22270259,  0.11246292,  0.32862678])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.erf(x)

    locals_var = locals().copy()
    kwargs = {}
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    return generate_layer_fn('erf')(**kwargs)
