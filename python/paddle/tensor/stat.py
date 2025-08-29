#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypeAlias, overload

import paddle
from paddle import _C_ops
from paddle.framework import (
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
)
from paddle.utils.decorator_utils import (
    ParamAliasDecorator,
    param_two_alias,
    param_two_alias_one_default,
)

from ..base.data_feeder import check_type, check_variable_and_dtype
from ..common_ops_import import Variable
from ..framework import LayerHelper, convert_np_dtype_to_dtype_, core
from .manipulation import cast
from .math import _get_reduce_axis_with_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle._typing import DTypeLike

_Interpolation: TypeAlias = Literal[
    'linear', 'higher', 'lower', 'midpoint', 'nearest'
]
__all__ = []


@param_two_alias(["x", "input"], ["axis", "dim"])
def mean(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    dtype: DTypeLike | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """
    Computes the mean of the input tensor's elements along ``axis``.

    Args:
        x (Tensor): The input Tensor with data type bool, bfloat16, float16, float32,
            float64, int32, int64, complex64, complex128.
            alias: ``input``
        axis (int|list|tuple|None, optional): The axis along which to perform mean
            calculations. ``axis`` should be int, list(int) or tuple(int). If
            ``axis`` is a list/tuple of dimension(s), mean is calculated along
            all element(s) of ``axis`` . ``axis`` or element(s) of ``axis``
            should be in range [-D, D), where D is the dimensions of ``x`` . If
            ``axis`` or element(s) of ``axis`` is less than 0, it works the
            same way as :math:`axis + D` . If ``axis`` is None, mean is
            calculated over all elements of ``x``. Default is None.
            alias: ``dim``
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
        dtype (str): The desired data type of returned tensor. Default: None.
        out(Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor, results of average along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[[1., 2., 3., 4.],
            ...                        [5., 6., 7., 8.],
            ...                        [9., 10., 11., 12.]],
            ...                       [[13., 14., 15., 16.],
            ...                        [17., 18., 19., 20.],
            ...                        [21., 22., 23., 24.]]])
            >>> out1 = paddle.mean(x)
            >>> print(out1.numpy())
            12.5
            >>> out2 = paddle.mean(x, axis=-1)
            >>> print(out2.numpy())
            [[ 2.5  6.5 10.5]
             [14.5 18.5 22.5]]
            >>> out3 = paddle.mean(x, axis=-1, keepdim=True)
            >>> print(out3.numpy())
            [[[ 2.5]
              [ 6.5]
              [10.5]]
             [[14.5]
              [18.5]
              [22.5]]]
            >>> out4 = paddle.mean(x, axis=[0, 2])
            >>> print(out4.numpy())
            [ 8.5 12.5 16.5]
            >>> out5 = paddle.mean(x, dtype='float64')
            >>> out5
            Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=True,
                12.50000000)
    """
    if dtype is not None:
        if not isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
            dtype = convert_np_dtype_to_dtype_(dtype)
        if x.dtype != dtype:
            x = cast(x, dtype)

    if in_dynamic_or_pir_mode():
        return _C_ops.mean(x, axis, keepdim, out=out)
    else:
        reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
        check_variable_and_dtype(
            x,
            'x/input',
            [
                'bool',
                'uint16',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'mean/reduce_mean',
        )
        check_type(
            axis, 'axis/dim', (int, list, tuple, Variable), 'mean/reduce_mean'
        )
        if isinstance(axis, (list, tuple)):
            for item in axis:
                check_type(
                    item,
                    'elements of axis/dim',
                    (int, Variable),
                    'mean/reduce_mean',
                )

        helper = LayerHelper('mean', **locals())

        attrs = {'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all}
        out_tensor = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='reduce_mean',
            inputs={'X': x},
            outputs={'Out': out_tensor},
            attrs=attrs,
        )
        return out_tensor


@ParamAliasDecorator({"x": ["input"], "axis": ["dim"]})
def var(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    unbiased: bool | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    correction: float = 1,
    out: Tensor | None = None,
) -> Tensor:
    """
    Computes the variance of ``x`` along ``axis`` .

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``, and ``dim`` can be used as an alias for ``axis``.
        For example, ``var(input=tensor_x, dim=1, ...)`` is equivalent to ``var(x=tensor_x, axis=1, ...)``.

    Args:
        x (Tensor): The input Tensor with data type float16, float32, float64.
            alias: ``input``.
        axis (int|list|tuple|None, optional): The axis along which to perform variance calculations. ``axis`` should be int, list(int) or tuple(int).
            alias: ``dim``.

            - If ``axis`` is a list/tuple of dimension(s), variance is calculated along all element(s) of ``axis`` . ``axis`` or element(s) of ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            - If ``axis`` or element(s) of ``axis`` is less than 0, it works the same way as :math:`axis + D` .
            - If ``axis`` is None, variance is calculated over all elements of ``x``. Default is None.

        unbiased (bool, optional): Whether to use the unbiased estimation. If ``unbiased`` is True, the divisor used in the computation is :math:`N - 1`, where :math:`N` represents the number of elements along ``axis`` , otherwise the divisor is :math:`N`. Default is True.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the output Tensor. The result tensor will have one fewer dimension than the input unless keep_dim is true. Default is False.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        correction (int|float, optional): Difference between the sample size and sample degrees of freedom.
            Defaults to 1 (Bessel's correction). If unbiased is specified, this parameter is ignored.
        out (Tensor|None, optional): Output tensor. Default is None.

    Returns:
        Tensor, results of variance along ``axis`` of ``x``, with the same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
            >>> out1 = paddle.var(x)
            >>> print(out1.numpy())
            2.6666667
            >>> out2 = paddle.var(x, axis=1)
            >>> print(out2.numpy())
            [1.         4.3333335]
    """
    if unbiased is not None and correction != 1:
        raise ValueError("Only one of unbiased and correction may be given")

    if unbiased is not None:
        actual_correction = 1.0 if unbiased else 0.0
    else:
        actual_correction = float(correction)
    if not in_dynamic_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'var'
        )

    u = mean(x, axis, True, name)
    dtype = paddle.float32 if x.dtype == paddle.float16 else x.dtype
    out_tensor = paddle.sum(
        paddle.pow((x - u), 2), axis, keepdim=keepdim, name=name, dtype=dtype
    )

    n = paddle.cast(paddle.numel(x), "int64") / paddle.cast(
        paddle.numel(out_tensor), "int64"
    )
    n = n.astype(dtype)

    if actual_correction != 0:
        corrected_n = n - actual_correction
        corrected_n = paddle.maximum(
            corrected_n, paddle.zeros_like(corrected_n)
        )
        if paddle.in_dynamic_mode() and paddle.any(corrected_n <= 0):
            warnings.warn("Degrees of freedom is <= 0.", stacklevel=2)
    else:
        corrected_n = n

    corrected_n.stop_gradient = True
    out_tensor /= corrected_n

    def _replace_nan(out):
        indices = paddle.arange(out.numel(), dtype='int64')
        out_nan = paddle.index_fill(
            out.flatten(), indices, 0, float('nan')
        ).reshape(out.shape)
        return out_nan

    if 0 in x.shape:
        out_tensor = _replace_nan(out_tensor)
    if len(x.shape) == 0 and actual_correction == 0:
        out_tensor = paddle.to_tensor(0, stop_gradient=out_tensor.stop_gradient)

    if out_tensor.dtype != x.dtype:
        result = out_tensor.astype(x.dtype)
    else:
        result = out_tensor

    if out is not None:
        paddle.assign(result, out)
        return out

    return result


def std(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    unbiased: bool = True,
    keepdim: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    Computes the standard-deviation of ``x`` along ``axis`` .

    Args:
        x (Tensor): The input Tensor with data type float16, float32, float64.
        axis (int|list|tuple|None, optional): The axis along which to perform
            standard-deviation calculations. ``axis`` should be int, list(int)
            or tuple(int). If ``axis`` is a list/tuple of dimension(s),
            standard-deviation is calculated along all element(s) of ``axis`` .
            ``axis`` or element(s) of ``axis`` should be in range [-D, D),
            where D is the dimensions of ``x`` . If ``axis`` or element(s) of
            ``axis`` is less than 0, it works the same way as :math:`axis + D` .
            If ``axis`` is None, standard-deviation is calculated over all
            elements of ``x``. Default is None.
        unbiased (bool, optional): Whether to use the unbiased estimation. If
            ``unbiased`` is True, the standard-deviation is calculated via the
            unbiased estimator. If ``unbiased`` is True,  the divisor used in
            the computation is :math:`N - 1`, where :math:`N` represents the
            number of elements along ``axis`` , otherwise the divisor is
            :math:`N`. Default is True.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of standard-deviation along ``axis`` of ``x``, with the
        same data type as ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
            >>> out1 = paddle.std(x)
            >>> print(out1.numpy())
            1.6329932
            >>> out2 = paddle.std(x, unbiased=False)
            >>> print(out2.numpy())
            1.490712
            >>> out3 = paddle.std(x, axis=1)
            >>> print(out3.numpy())
            [1.       2.081666]

    """
    if not in_dynamic_or_pir_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'std'
        )
    out = var(**locals())
    return paddle.sqrt(out)


def numel(x: Tensor, name: str | None = None) -> Tensor:
    """
    Returns the number of elements for a tensor, which is a 0-D int64 Tensor with shape [].

    Args:
        x (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, uint8, int8, int32, int64, complex64, complex128.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The number of elements for the input Tensor, whose shape is [].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
            >>> numel = paddle.numel(x)
            >>> print(numel.numpy())
            140


    """
    if in_dynamic_or_pir_mode():
        return _C_ops.numel(x)
    else:
        if not isinstance(x, Variable):
            raise TypeError("x must be a Tensor in numel")
        helper = LayerHelper('numel', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.INT64
        )
        helper.append_op(type='size', inputs={'Input': x}, outputs={'Out': out})
        return out


@overload
def nanmedian(
    x: Tensor,
    axis: int,
    keepdim: bool = ...,
    mode: Literal['min'] = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def nanmedian(
    x: Tensor,
    axis: int | Sequence[int] | None = ...,
    keepdim: bool = ...,
    mode: Literal['avg', 'min'] = ...,
    name: str | None = ...,
) -> Tensor: ...


def nanmedian(
    x,
    axis=None,
    keepdim=False,
    mode='avg',
    name=None,
):
    r"""
    Compute the median along the specified axis, while ignoring NaNs.

    If the valid count of elements is a even number,
    the average value of both elements in the middle is calculated as the median.

    Args:
        x (Tensor): The input Tensor, it's data type can be int32, int64, float16, bfloat16, float32, float64.
        axis (None|int|list|tuple, optional):
            The axis along which to perform median calculations ``axis`` should be int or list of int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is None, median is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        mode (str, optional): Whether to use mean or min operation to calculate
            the nanmedian values when the input tensor has an even number of non-NaN elements
            along the dimension ``axis``. Support 'avg' and 'min'. Default is 'avg'.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor or tuple of Tensor. If ``mode`` == 'min' and ``axis`` is int, the result
        will be a tuple of two tensors (nanmedian value and nanmedian index). Otherwise,
        only nanmedian value will be returned.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[float('nan'), 2. , 3. ], [0. , 1. , 2. ]])

            >>> y1 = x.nanmedian()
            >>> print(y1.numpy())
            2.0

            >>> y2 = x.nanmedian(0)
            >>> print(y2.numpy())
            [0.  1.5 2.5]

            >>> y3 = x.nanmedian(0, keepdim=True)
            >>> print(y3.numpy())
            [[0.  1.5 2.5]]

            >>> y4 = x.nanmedian((0, 1))
            >>> print(y4.numpy())
            2.0

            >>> y5 = x.nanmedian(mode='min')
            >>> print(y5.numpy())
            2.0

            >>> y6, y6_index = x.nanmedian(0, mode='min')
            >>> print(y6.numpy())
            [0. 1. 2.]
            >>> print(y6_index.numpy())
            [1 1 1]

            >>> y7, y7_index = x.nanmedian(1, mode='min')
            >>> print(y7.numpy())
            [2. 1.]
            >>> print(y7_index.numpy())
            [1 1]

            >>> y8 = x.nanmedian((0,1), mode='min')
            >>> print(y8.numpy())
            2.0
    """
    if not isinstance(x, (Variable, paddle.pir.Value)):
        raise TypeError("In median, the input x should be a Tensor.")

    if isinstance(axis, (list, tuple)) and len(axis) == 0:
        raise ValueError("Axis list should not be empty.")

    if mode not in ('avg', 'min'):
        raise ValueError(f"Mode {mode} is not supported. Must be avg or min.")

    need_index = (axis is not None) and (not isinstance(axis, (list, tuple)))
    if axis is None:
        axis = []
    elif isinstance(axis, tuple):
        axis = list(axis)
    elif isinstance(axis, int):
        axis = [axis]

    if in_dynamic_or_pir_mode():
        out, indices = _C_ops.nanmedian(x, axis, keepdim, mode)
        indices.stop_gradient = True
    else:
        check_variable_and_dtype(
            x,
            'X',
            ['int32', 'int64', 'float16', 'float32', 'float64', 'uint16'],
            'nanmedian',
        )

        helper = LayerHelper('nanmedian', **locals())
        attrs = {'axis': axis, 'keepdim': keepdim, 'mode': mode}
        out = helper.create_variable_for_type_inference(x.dtype)
        indices = helper.create_variable_for_type_inference(paddle.int64)
        helper.append_op(
            type='nanmedian',
            inputs={'X': x},
            outputs={'Out': out, 'MedianIndex': indices},
            attrs=attrs,
        )
        indices.stop_gradient = True
    if mode == 'min' and need_index:
        return out, indices
    else:
        return out


@overload
def median(
    x: Tensor,
    axis: int = ...,
    keepdim: bool = ...,
    mode: Literal['min'] = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def median(
    x: Tensor,
    axis: int | None = ...,
    keepdim: bool = ...,
    mode: Literal['avg', 'min'] = ...,
    name: str | None = ...,
) -> Tensor: ...


@param_two_alias_one_default(["x", "input"], ["axis", "dim"], ["mode", 'min'])
def median(
    x,
    axis=None,
    keepdim=False,
    mode='avg',
    name=None,
):
    """
    Compute the median along the specified axis.

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``, and ``dim`` can be used as an alias for ``axis``.
        When an alias replacement occurs, the default parameter for mode setting is min instead of avg.
        For example, ``median(input=tensor_x, dim=1, ...)`` is equivalent to ``median(x=tensor_x, axis=1, ...)``.

    Args:
        x (Tensor): The input Tensor, it's data type can be bfloat16, float16, float32, float64, int32, int64.
            alias: ``input``.
        axis (int|None, optional): The axis along which to perform median calculations ``axis`` should be int.
            alias: ``dim``.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is None, median is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        mode (str, optional): Whether to use mean or min operation to calculate
            the median values when the input tensor has an even number of elements
            in the dimension ``axis``. Support 'avg' and 'min'. Default is 'avg'.
            When an alias replacement occurs, the default parameter for mode setting is min instead of avg.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor or tuple of Tensor.
        If ``mode`` == 'avg', the result will be the tensor of median values;
        If ``mode`` == 'min' and ``axis`` is None, the result will be the tensor of median values;
        If ``mode`` == 'min' and ``axis`` is not None, the result will be a tuple of two tensors
        containing median values and their indices.

        When ``mode`` == 'avg', if data type of ``x`` is float64, data type of median values will be float64,
        otherwise data type of median values will be float32.
        When ``mode`` == 'min', the data type of median values will be the same as ``x``. The data type of
        indices will be int64.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> x = paddle.arange(12).reshape([3, 4])
            >>> print(x)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0 , 1 , 2 , 3 ],
             [4 , 5 , 6 , 7 ],
             [8 , 9 , 10, 11]])

            >>> y1 = paddle.median(x)
            >>> print(y1)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            5.50000000)

            >>> y2 = paddle.median(x, axis=0)
            >>> print(y2)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [4., 5., 6., 7.])

            >>> y3 = paddle.median(x, axis=1)
            >>> print(y3)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.50000000, 5.50000000, 9.50000000])

            >>> y4 = paddle.median(x, axis=0, keepdim=True)
            >>> print(y4)
            Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[4., 5., 6., 7.]])

            >>> y5 = paddle.median(x, mode='min')
            >>> print(y5)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            5)

            >>> median_value, median_indices = paddle.median(x, axis=1, mode='min')
            >>> print(median_value)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 5, 9])
            >>> print(median_indices)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 1, 1])

            >>> # cases containing nan values
            >>> x = paddle.to_tensor(np.array([[1,float('nan'),3,float('nan')],[1,2,3,4],[float('nan'),1,2,3]]))

            >>> y6 = paddle.median(x, axis=-1, keepdim=True)
            >>> print(y6)
            Tensor(shape=[3, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[nan       ],
             [2.50000000],
             [nan       ]])

            >>> median_value, median_indices = paddle.median(x, axis=1, keepdim=True, mode='min')
            >>> print(median_value)
            Tensor(shape=[3, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[nan],
             [2. ],
             [nan]])
            >>> print(median_indices)
            Tensor(shape=[3, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1],
             [1],
             [0]])
    """
    if not isinstance(x, (Variable, paddle.pir.Value)):
        raise TypeError("In median, the input x should be a Tensor.")

    if isinstance(axis, (list, tuple)) and len(axis) == 0:
        raise ValueError("Axis list should not be empty.")
    dims = len(x.shape)
    if dims == 0:
        assert axis in [
            -1,
            0,
            None,
        ], 'when input 0-D, axis can only be [-1, 0] or default None'
    elif axis is not None:
        if not isinstance(axis, int) or not (axis < dims and axis >= -dims):
            raise ValueError(
                "In median, axis should be none or an integer in range [-rank(x), rank(x))."
            )

    if mode not in ('avg', 'min'):
        raise ValueError(f"Mode {mode} is not supported. Must be avg or min.")
    need_idx = axis is not None
    if axis is None:
        is_flatten = True

    if axis is None:
        axis = []
    elif isinstance(axis, int):
        axis = [axis]

    if mode == "avg" and not x.dtype == paddle.float64:
        x = x.astype(paddle.float32)

    out, indices = _C_ops.median(x, axis, keepdim, mode)
    indices.stop_gradient = True

    if mode == 'min' and need_idx:
        return out, indices
    else:
        return out


def _compute_quantile(
    x: Tensor,
    q: float | Sequence[float] | Tensor | None,
    axis: int | list[int] | None = None,
    keepdim: bool = False,
    interpolation: _Interpolation = "linear",
    ignore_nan: bool = False,
) -> Tensor:
    """
    Compute the quantile of the input along the specified axis.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64, int32, int64.
        q (int|float|list|Tensor): The q for calculate quantile, which should be in range [0, 1]. If q is a list or
            a 1-D Tensor, each element of q will be calculated and the first dimension of output is same to the number of ``q`` .
            If q is a 0-D Tensor, it will be treated as an integer or float.
        axis (int|list, optional): The axis along which to calculate quantile. ``axis`` should be int or list of int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is a list, quantile is calculated over all elements of given axes.
            If ``axis`` is None, quantile is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        interpolation (str, optional): The interpolation method to use
            when the desired quantile falls between two data points. Must be one of linear, higher,
            lower, midpoint and nearest. Default is linear.
        ignore_nan: (bool, optional): Whether to ignore NaN of input Tensor.
            If ``ignore_nan`` is True, it will calculate nanquantile.
            Otherwise it will calculate quantile. Default is False.

    Returns:
        Tensor, results of quantile along ``axis`` of ``x``.
        In order to obtain higher precision, data type of results will be float64.
    """
    # Validate x
    if not isinstance(x, (Variable, paddle.pir.Value)):
        raise TypeError("input x should be a Tensor.")

    # Validate q
    if isinstance(q, (int, float)):
        q = [q]
    elif isinstance(q, (list, tuple)):
        if len(q) <= 0:
            raise ValueError("q should not be empty")
    elif isinstance(q, (Variable, paddle.pir.Value)):
        if len(q.shape) > 1:
            raise ValueError("q should be a 0-D tensor or a 1-D tensor")
        if len(q.shape) == 0:
            q = [q]
    else:
        raise TypeError(
            "Type of q should be int, float, list or tuple, or tensor"
        )
    for q_num in q:
        # we do not validate tensor q in static mode
        if not in_dynamic_mode() and isinstance(
            q_num, (Variable, paddle.pir.Value)
        ):
            break
        if q_num < 0 or q_num > 1:
            raise ValueError("q should be in range [0, 1]")

    if interpolation not in [
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
    ]:
        raise ValueError(
            f"interpolation must be one of 'linear', 'lower', 'higher', 'nearest' or 'midpoint', but got {interpolation}"
        )
    # Validate axis
    dims = len(x.shape)
    out_shape = list(x.shape)
    if axis is None:
        x = paddle.flatten(x)
        axis = 0
        out_shape = [1] * dims
    else:
        if isinstance(axis, list):
            axis_src, axis_dst = [], []
            for axis_single in axis:
                if not isinstance(axis_single, int) or not (
                    axis_single < dims and axis_single >= -dims
                ):
                    raise ValueError(
                        "Axis should be None, int, or a list, element should in range [-rank(x), rank(x))."
                    )
                if axis_single < 0:
                    axis_single = axis_single + dims
                axis_src.append(axis_single)
                out_shape[axis_single] = 1

            axis_dst = list(range(-len(axis), 0))
            x = paddle.moveaxis(x, axis_src, axis_dst)
            if len(axis_dst) == 0:
                x = paddle.flatten(x)
                axis = 0
            else:
                x = paddle.flatten(x, axis_dst[0], axis_dst[-1])
                axis = axis_dst[0]
        else:
            if not isinstance(axis, int) or not (axis < dims and axis >= -dims):
                raise ValueError(
                    "Axis should be None, int, or a list, element should in range [-rank(x), rank(x))."
                )
            if axis < 0:
                axis += dims
            out_shape[axis] = 1

    mask = x.isnan()
    valid_counts = mask.logical_not().sum(axis=axis, keepdim=True)

    indices = []

    for q_num in q:
        if in_dynamic_or_pir_mode():
            q_num = paddle.to_tensor(q_num, dtype=x.dtype)
        if ignore_nan:
            indices.append(q_num * (valid_counts - 1))
        else:
            index = q_num * (valid_counts - 1)
            last_index = x.shape[axis] - 1
            nums = paddle.full_like(index, fill_value=last_index)
            index = paddle.where(mask.any(axis=axis, keepdim=True), nums, index)
            indices.append(index)

    sorted_tensor = paddle.sort(x, axis)

    def _compute_index(index):
        if interpolation == "nearest":
            idx = paddle.round(index).astype(paddle.int32)
            return paddle.take_along_axis(sorted_tensor, idx, axis=axis)

        indices_below = paddle.floor(index).astype(paddle.int32)
        if interpolation != "higher":
            # avoid unnecessary compute
            tensor_below = paddle.take_along_axis(
                sorted_tensor, indices_below, axis=axis
            )
        if interpolation == "lower":
            return tensor_below

        indices_upper = paddle.ceil(index).astype(paddle.int32)
        tensor_upper = paddle.take_along_axis(
            sorted_tensor, indices_upper, axis=axis
        )
        if interpolation == "higher":
            return tensor_upper

        if interpolation == "midpoint":
            return (tensor_upper + tensor_below) / 2

        weights = (index - indices_below.astype(index.dtype)).astype(x.dtype)
        # "linear"
        return paddle.lerp(
            tensor_below.astype(x.dtype),
            tensor_upper.astype(x.dtype),
            weights,
        )

    outputs = []

    # TODO(chenjianye): replace the for-loop to directly take elements.
    for index in indices:
        out = _compute_index(index)
        if not keepdim:
            out = paddle.squeeze(out, axis=axis)
        else:
            out = out.reshape(out_shape)
        outputs.append(out)

    if len(outputs) > 1:
        outputs = paddle.stack(outputs, 0)
    else:
        outputs = outputs[0]
    # return outputs.astype(x.dtype)
    return outputs


def quantile(
    x: Tensor,
    q: float | Sequence[float] | Tensor,
    axis: int | list[int] | None = None,
    keepdim: bool = False,
    interpolation: _Interpolation = "linear",
) -> Tensor:
    """
    Compute the quantile of the input along the specified axis.
    If any values in a reduced row are NaN, then the quantiles for that reduction will be NaN.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64, int32, int64.
        q (int|float|list|Tensor): The q for calculate quantile, which should be in range [0, 1]. If q is a list or
            a 1-D Tensor, each element of q will be calculated and the first dimension of output is same to the number of ``q`` .
            If q is a 0-D Tensor, it will be treated as an integer or float.
        axis (int|list, optional): The axis along which to calculate quantile. ``axis`` should be int or list of int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is a list, quantile is calculated over all elements of given axes.
            If ``axis`` is None, quantile is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        interpolation (str, optional): The interpolation method to use
            when the desired quantile falls between two data points. Must be one of linear, higher,
            lower, midpoint and nearest. Default is linear.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of quantile along ``axis`` of ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> y = paddle.arange(0, 8 ,dtype="float32").reshape([4, 2])
            >>> print(y)
            Tensor(shape=[4, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 1.],
             [2., 3.],
             [4., 5.],
             [6., 7.]])

            >>> y1 = paddle.quantile(y, q=0.5, axis=[0, 1])
            >>> print(y1)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            3.50000000)

            >>> y2 = paddle.quantile(y, q=0.5, axis=1)
            >>> print(y2)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.50000000, 2.50000000, 4.50000000, 6.50000000])

            >>> y3 = paddle.quantile(y, q=[0.3, 0.5], axis=0)
            >>> print(y3)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.80000000, 2.80000000],
             [3.        , 4.        ]])

            >>> y[0,0] = float("nan")
            >>> y4 = paddle.quantile(y, q=0.8, axis=1, keepdim=True)
            >>> print(y4)
            Tensor(shape=[4, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[nan       ],
             [2.80000000],
             [4.80000000],
             [6.80000000]])

    """
    return _compute_quantile(
        x,
        q,
        axis=axis,
        keepdim=keepdim,
        interpolation=interpolation,
        ignore_nan=False,
    )


def nanquantile(
    x: Tensor,
    q: float | Sequence[float] | Tensor,
    axis: list[int] | int | None = None,
    keepdim: bool = False,
    interpolation: _Interpolation = "linear",
) -> Tensor:
    """
    Compute the quantile of the input as if NaN values in input did not exist.
    If all values in a reduced row are NaN, then the quantiles for that reduction will be NaN.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64, int32, int64.
        q (int|float|list|Tensor): The q for calculate quantile, which should be in range [0, 1]. If q is a list or
            a 1-D Tensor, each element of q will be calculated and the first dimension of output is same to the number of ``q`` .
            If q is a 0-D Tensor, it will be treated as an integer or float.
        axis (int|list, optional): The axis along which to calculate quantile. ``axis`` should be int or list of int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is a list, quantile is calculated over all elements of given axes.
            If ``axis`` is None, quantile is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        interpolation (str, optional): The interpolation method to use
            when the desired quantile falls between two data points. Must be one of linear, higher,
            lower, midpoint and nearest. Default is linear.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of quantile along ``axis`` of ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor(
            ...     [[0, 1, 2, 3, 4],
            ...      [5, 6, 7, 8, 9]],
            ...     dtype="float32")
            >>> x[0,0] = float("nan")

            >>> y1 = paddle.nanquantile(x, q=0.5, axis=[0, 1])
            >>> print(y1)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            5.)

            >>> y2 = paddle.nanquantile(x, q=0.5, axis=1)
            >>> print(y2)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2.50000000, 7.        ])

            >>> y3 = paddle.nanquantile(x, q=[0.3, 0.5], axis=0)
            >>> print(y3)
            Tensor(shape=[2, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[5.        , 2.50000000, 3.50000000, 4.50000000, 5.50000000],
             [5.        , 3.50000000, 4.50000000, 5.50000000, 6.50000000]])

            >>> y4 = paddle.nanquantile(x, q=0.8, axis=1, keepdim=True)
            >>> print(y4)
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[3.40000000],
             [8.20000000]])

            >>> nan = paddle.full(shape=[2, 3], fill_value=float("nan"))
            >>> y5 = paddle.nanquantile(nan, q=0.8, axis=1, keepdim=True)
            >>> print(y5)
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[nan],
             [nan]])

    """
    return _compute_quantile(
        x,
        q,
        axis=axis,
        keepdim=keepdim,
        interpolation=interpolation,
        ignore_nan=True,
    )
