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
    func_name="paddle.compat.softmax",
    correct_name="paddle.nn.functional.softmax",
)
def softmax(
    input: Tensor,
    dim: int | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    This operator implements the compat.softmax. The calculation process is as follows:

    1. The dimension :attr:`dim` of ``input`` will be permuted to the last.

    2. Then ``input`` will be logically flattened to a 2-D matrix. The matrix's second
    dimension(row length) is the same as the dimension :attr:`axis` of ``input``,
    and the first dimension(column length) is the product of all other dimensions
    of ``input``. For each row of the matrix, the softmax operator squashes the
    K-dimensional(K is the width of the matrix, which is also the size of ``input``'s
    dimension :attr:`dim`) vector of arbitrary real values to a K-dimensional
    vector of real values in the range [0, 1] that add up to 1.

    3. After the softmax operation is completed, the inverse operations of steps 1 and 2
    are performed to restore the two-dimensional matrix to the same dimension as the ``input`` .

    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.

    For each row :math:`i` and each column :math:`j` in the matrix, we have:

    .. math::

        softmax[i, j] = \frac{\exp(input[i, j])}{\sum_j(exp(input[i, j])}

    Example:

    .. code-block:: text

        Case 1:
          Input:
            input.shape = [2, 3, 4]
            input.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]

          Attrs:
            dim = -1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                        [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]

        Case 2:
          Input:
            input.shape = [2, 3, 4]
            input.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]
          Attrs:
            dim = 1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
                         [0.01786798, 0.01786798, 0.04661262, 0.04661262],
                         [0.97555875, 0.97555875, 0.93623955, 0.93623955]],
                        [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
                         [0.26762315, 0.26762315, 0.26762315, 0.26762315],
                         [0.72747516, 0.72747516, 0.72747516, 0.72747516]]]

    Parameters:
        input (Tensor): The input Tensor with data type bfloat16, float16, float32, float64.
        dim (int, optional): The dim along which to perform softmax
            calculations. It should be in range [-D, D), where D is the
            rank of ``input`` . If ``dim`` < 0, it works the same way as
            :math:`dim + D` . Default is None.
        dtype (str, optional): The data type of the output tensor, can be bfloat16, float16, float32, float64.
        out (Tensor, optional): The output Tensor.

    Returns:
        A Tensor with the same shape and data type (use ``dtype`` if it is
        specified) as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[[2.0, 3.0, 4.0, 5.0],
            ...                        [3.0, 4.0, 5.0, 6.0],
            ...                        [7.0, 8.0, 8.0, 9.0]],
            ...                       [[1.0, 2.0, 3.0, 4.0],
            ...                        [5.0, 6.0, 7.0, 8.0],
            ...                        [6.0, 7.0, 8.0, 9.0]]],dtype='float32')
            >>> out1 = paddle.compat.softmax(x, -1)
            >>> out2 = paddle.compat.softmax(x, -1, dtype='float64')
            >>> #out1's data type is float32; out2's data type is float64
            >>> #out1 and out2's value is as follows:
            >>> print(out1)
            >>> print(out2)
            Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.07232949, 0.19661194, 0.19661194, 0.53444666]],
             [[0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428],
              [0.03205860, 0.08714432, 0.23688284, 0.64391428]]])
            Tensor(shape=[2, 3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[[0.03205860, 0.08714432, 0.23688282, 0.64391426],
              [0.03205860, 0.08714432, 0.23688282, 0.64391426],
              [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
             [[0.03205860, 0.08714432, 0.23688282, 0.64391426],
              [0.03205860, 0.08714432, 0.23688282, 0.64391426],
              [0.03205860, 0.08714432, 0.23688282, 0.64391426]]])
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
        return _C_ops.softmax(outs_cast, dim, out=out)
