#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import _C_ops
from paddle.framework import core, in_dynamic_or_pir_mode
from paddle.utils.decorator_utils import ForbidKeywordsIgnoreOneParamDecorator

from ..base.framework import convert_np_dtype_to_dtype_

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import DTypeLike


@ForbidKeywordsIgnoreOneParamDecorator(
    illegal_keys={"x", "axis", "name"},
    ignore_param=('_stacklevel', 2, int),
    func_name="paddle.compat.nn.functional.log_softmax",
    correct_name="paddle.nn.functional.log_softmax",
    url_suffix="torch.nn.functional.log_softmax",
)
def log_softmax(
    input: Tensor,
    dim: int | None = None,
    dtype: DTypeLike | None = None,
) -> Tensor:
    r"""
    This operator implements PyTorch compatible log_softmax. The calculation process is as follows:

    .. math::

        \begin{aligned}
        log\_softmax[i, j] &= log(softmax(input)) \\
        &= log\left(\frac{\exp(input[i, j])}{\sum_j \exp(input[i, j])}\right)
        \end{aligned}

    Parameters:
        input (Tensor): The input Tensor with data type float32, float64.
        dim (int, optional): The dim along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            rank of ``input``. If ``dim`` < 0, it works the same way as
            :math:`dim + D`. If ``dim`` is None, it defaults to 0 for
            0-D, 1-D, and 3-D tensors, and 1 for 2-D tensors (same as
            PyTorch behavior). Default is None.
        dtype (str|np.dtype|core.VarDesc.VarType|core.DataType, optional):
            The desired data type of the output tensor. If dtype is
            specified, ``input`` is cast to ``dtype`` before the operation
            is performed. Supported dtype: float32, float64. If ``dtype``
            is None, the output Tensor has the same dtype as input.
            Default is None.

    Returns:
        A Tensor with the same shape and data type (use ``dtype`` if it is
        specified) as input.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor(
            ...     [
            ...         [[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 8.0, 9.0]],
            ...         [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
            ...     ],
            ...     dtype='float32',
            ... )
            >>> out1 = paddle.compat.nn.functional.log_softmax(x, -1)
            >>> out2 = paddle.compat.nn.functional.log_softmax(x, -1, dtype='float64')
            >>> # out1's data type is float32; out2's data type is float64
            >>> print(out1)
            >>> print(out2)
    """
    if dim is None:
        ndim = input.ndim
        if ndim == 0 or ndim == 1 or ndim == 3:
            dim = 0
        else:
            dim = 1

    if (
        (dtype is not None)
        and (not isinstance(dtype, core.VarDesc.VarType))
        and (not isinstance(dtype, core.DataType))
    ):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_or_pir_mode():
        outs_cast = input if dtype is None else _C_ops.cast(input, dtype)
        return _C_ops.log_softmax(outs_cast, dim)
