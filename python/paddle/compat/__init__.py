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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from typing_extensions import overload

import paddle
from paddle import _C_ops
from paddle.base import core
from paddle.base.framework import Variable
from paddle.framework import (
    in_dynamic_mode,
)
from paddle.utils.decorator_utils import ForbidKeywordsDecorator

from . import nn as nn
from .proxy import (  # noqa: F401
    disable_torch_proxy,
    enable_torch_proxy,
    extend_torch_proxy_blocked_modules,
    paddle_triton_fun,
    use_torch_proxy_guard,
)
from .utils import _check_out_status

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor

__all__ = [
    'equal',
    'slogdet',
    'sort',
    'split',
    'min',
    'max',
    'unique',
    'median',
    'nanmedian',
    'seed',
]


def __getattr__(name):
    if name == "paddle_triton":
        return paddle_triton_fun()


_types = [
    paddle.uint8,
    paddle.int8,
    paddle.int16,
    paddle.int32,
    paddle.int64,
    paddle.float16,
    paddle.float32,
    paddle.float64,
    paddle.bool,
    paddle.bfloat16,
]
u1, i1, i2, i4, i8, f2, f4, f8, b1, bf = _types

_promote_matrix = [
    [u1, i2, i2, i4, i8, f2, f4, f8, u1, bf],  # u1
    [i2, i1, i2, i4, i8, f2, f4, f8, i1, bf],  # i1
    [i2, i2, i2, i4, i8, f2, f4, f8, i2, bf],  # i2
    [i4, i4, i4, i4, i8, f2, f4, f8, i4, bf],  # i4
    [i8, i8, i8, i8, i8, f2, f4, f8, i8, bf],  # i8
    [f2, f2, f2, f2, f2, f2, f4, f8, f2, f4],  # f2
    [f4, f4, f4, f4, f4, f4, f4, f8, f4, f4],  # f4
    [f8, f8, f8, f8, f8, f8, f8, f8, f8, f8],  # f8
    [u1, i1, i2, i4, i8, f2, f4, f8, b1, bf],  # b1
    [bf, bf, bf, bf, bf, f4, f4, f8, bf, bf],  # bf
]

PROMOTE_DICT = {
    (t1, t2): _promote_matrix[i][j]
    for i, t1 in enumerate(_types)
    for j, t2 in enumerate(_types)
}


@ForbidKeywordsDecorator(
    illegal_keys={"x", "y"},
    func_name="paddle.compat.equal",
    correct_name="paddle.equal",
)
def equal(
    input: Tensor,
    other: Tensor,
) -> bool:
    """

    ``True`` if two tensors have the same size and elements, ``False`` otherwise.

    Note:
        Tensors containing NaNs are never equal to each other. Additionally, this function does not differentiate between the data types of the tensors during comparison.

    Args:
        input (Tensor): Tensor, data type is bool, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
        other (Tensor): Tensor, data type is bool, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.

    Returns:
        Bool: ``True`` if two tensors have the same size and elements, ``False`` otherwise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.compat.equal(x, y)
            >>> print(result1)
            False
    """
    if input.dtype == other.dtype:
        return paddle.equal_all(input, other).item()

    common_dtype = PROMOTE_DICT.get(input.dtype, other.dtype)
    if input.dtype != common_dtype:
        input = input.cast(common_dtype)
    if other.dtype != common_dtype:
        other = other.cast(common_dtype)

    return paddle.equal_all(input, other).item()


class MedianRetType(NamedTuple):
    values: Tensor
    indices: Tensor


@ForbidKeywordsDecorator(
    illegal_keys={"x", "axis"},
    func_name="paddle.compat.median",
    correct_name="paddle.median",
)
def median(
    input: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    out: tuple[Tensor, Tensor] | Tensor | None = None,
) -> Tensor | MedianRetType:
    """
    Returns the median of the values in input.

    Args:
        input (Tensor): The input tensor.
        dim (int|None, optional): The dimension to reduce. If None, computes the median over all elements. Default is None.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default is False.
        out (Tensor|tuple[Tensor, Tensor], optional): If provided, the result will be written into this tensor.
            For global median (dim=None), out must be a single tensor.
            For median along a dimension (dim specified, including dim=-1), out must be a tuple of two tensors (values, indices).

    Returns:
        Tensor|MedianRetType: If dim is None, returns a single tensor. If dim is specified (including dim=-1),
        returns a named tuple MedianRetType(values: Tensor, indices: Tensor).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> result = paddle.compat.median(x)
            >>> print(result)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True, 5)

            >>> ret = paddle.compat.median(x, dim=1)
            >>> print(ret.values)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [2, 5, 8])
            >>> print(ret.indices)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [1, 1, 1])

            >>> # Using out parameter
            >>> out_values = paddle.zeros([3], dtype='int64')
            >>> out_indices = paddle.zeros([3], dtype='int64')
            >>> paddle.compat.median(x, dim=1, out=(out_values, out_indices))
            >>> print(out_values)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [2, 5, 8])
    """
    if dim is None:
        _check_out_status(out, False)
        result = paddle.median(input, axis=dim, keepdim=keepdim, mode='min')
        if out is not None:
            paddle.assign(result, out)
            return out
        return result
    else:
        _check_out_status(out, True)
        values, indices = paddle.median(
            input, axis=dim, keepdim=keepdim, mode='min', out=out
        )
        if out is not None:
            return MedianRetType(values=out[0], indices=out[1])
        return MedianRetType(values=values, indices=indices)


@ForbidKeywordsDecorator(
    illegal_keys={"x", "axis"},
    func_name="paddle.compat.nanmedian",
    correct_name="paddle.nanmedian",
)
def nanmedian(
    input: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    out: tuple[Tensor, Tensor] | Tensor | None = None,
) -> Tensor | MedianRetType:
    """
    Returns the median of the values in input, ignoring NaN values.

    Args:
        input (Tensor): The input tensor.
        dim (int|None, optional): The dimension to reduce. If None, computes the nanmedian over all elements. Default is None.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default is False.
        out (Tensor|tuple[Tensor, Tensor], optional): If provided, the result will be written into this tensor.
            For global nanmedian (dim=None), out must be a single tensor.
            For nanmedian along a dimension (dim specified, including dim=-1), out must be a tuple of two tensors (values, indices).

    Returns:
        Tensor|MedianRetType: The median values, ignoring NaN. If dim is None, returns a single tensor. If dim is specified (including dim=-1),
        returns a named tuple MedianRetType(values: Tensor, indices: Tensor).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> x = paddle.to_tensor([[1, float('nan'), 3], [4, 5, 6], [float('nan'), 8, 9]], dtype='float32')
            >>> result = paddle.compat.nanmedian(x)
            >>> print(result)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True, 5.0)

            >>> ret = paddle.compat.nanmedian(x, dim=1)
            >>> print(ret.values)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True, [1.0, 5.0, 8.0])
            >>> print(ret.indices)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [0, 1, 1])

            >>> # Using out parameter
            >>> out_values = paddle.zeros([3], dtype='float32')
            >>> out_indices = paddle.zeros([3], dtype='int64')
            >>> paddle.compat.nanmedian(x, dim=1, out=(out_values, out_indices))
            >>> print(out_values)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True, [1.0, 5.0, 8.0])
    """
    if dim is None:
        _check_out_status(out, False)
        result = paddle.nanmedian(input, axis=dim, keepdim=keepdim, mode='min')
        if out is not None:
            paddle.assign(result, out)
            return out
        return result
    else:
        _check_out_status(out, True)
        values, indices = paddle.nanmedian(
            input, axis=dim, keepdim=keepdim, mode='min'
        )
        # This conversion is needed because PyTorch returns index 0 for all-nan rows,
        # while PaddlePaddle returns index -1 for all-nan rows
        indices = paddle.maximum(indices, paddle.zeros_like(indices))

        if out is not None:
            paddle.assign(values, out[0])
            paddle.assign(indices, out[1])
            return MedianRetType(values=out[0], indices=out[1])
        return MedianRetType(values=values, indices=indices)


def seed() -> int:
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number on all devices. Returns a 64 bit number used to seed the RNG.
    Returns:
        Returns: int64, the seed used to seed the RNG.
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> seed = paddle.compat.seed()
    """
    seed = core.default_cpu_generator().seed()
    paddle.seed(seed)
    return seed


class MinMaxRetType(NamedTuple):
    values: Tensor
    indices: Tensor


def _min_max_param_checker(func_name: str, *args: Any, **kwargs: Any):
    def invalid_arguments_exception(error_prefix=""):
        type_strs = [type(v).__name__ for v in args]
        type_strs.extend([f"{k}={type(v).__name__}" for k, v in kwargs.items()])
        signature = ", ".join(type_strs)

        error_msg = (
            f"Invalid arguments for `paddle.compat.{func_name}`:\n{error_prefix}"
            f"Got: (paddle.Tensor input, {signature}), but expect one of:\n"
            f" - (input: paddle.Tensor) for reduce_{func_name} on all dims.\n"
            f" - (input: paddle.Tensor, other: paddle.Tensor) -> see paddle.{func_name}imum\n"
            f" - (input: paddle.Tensor, int dim (cannot be None), bool keepdim = False)\n"
        )
        return TypeError(error_msg)

    def try_get_keys(key):
        res = None
        try:
            res = kwargs[key]
        except KeyError:
            raise invalid_arguments_exception() from None
        return res

    dim_or_other = None
    keepdim = False

    num_args = len(args)
    total_arg_num = num_args + len(kwargs)
    if total_arg_num > 2:
        raise invalid_arguments_exception()
    elif total_arg_num == 2:
        if num_args == 2:
            dim_or_other, keepdim = args
        elif num_args == 1:
            dim_or_other = args[0]
            keepdim = try_get_keys("keepdim")
        else:
            dim_or_other = try_get_keys("dim")
            keepdim = try_get_keys("keepdim")
        if dim_or_other is None or isinstance(
            dim_or_other, (Variable, paddle.pir.Value)
        ):
            raise invalid_arguments_exception()
    elif total_arg_num == 1:
        if num_args:
            dim_or_other = args[0]
        else:
            if "dim" in kwargs:
                dim_or_other = kwargs["dim"]
            elif "other" in kwargs:
                dim_or_other = kwargs["other"]
                if not isinstance(dim_or_other, (Variable, paddle.pir.Value)):
                    raise invalid_arguments_exception()
        if dim_or_other is None:
            raise invalid_arguments_exception()

    if (
        dim_or_other is not None
        and not isinstance(dim_or_other, (Variable, paddle.pir.Value))
        and type(dim_or_other) is not int
    ):
        raise invalid_arguments_exception(
            f"The second input must be int or Tensor or implicit None in compat.{func_name}, but received {type(dim_or_other)}.\n"
        )

    return dim_or_other, keepdim


def _min_max_tensor_allow_grad(input: Tensor):
    """Prevent integral input tensor type to have `stop_gradient=False`"""
    in_dtype = input.dtype
    if (
        in_dtype == paddle.int32
        or in_dtype == paddle.int64
        or in_dtype == paddle.uint8
        or in_dtype == paddle.int16
    ):
        if not input.stop_gradient:
            raise TypeError(
                f"Tensors with integral type: '{in_dtype}' should stop gradient."
            )


def _min_max_allow_cpu_composite(input: Tensor):
    """paddle.min/argmin(max/argmax), paddle.take_along_axis reject the following types"""
    in_dtype = input.dtype
    if (
        in_dtype == paddle.float16
        or in_dtype == paddle.bfloat16
        or in_dtype == paddle.int16
    ):
        raise TypeError(
            f"Non-CUDA GPU placed Tensor does not have '{in_dtype}' op registered.\n"
            "Paddle support following DataTypes: int32, int64, float64, float32, uint8"
        )


@ForbidKeywordsDecorator(
    illegal_keys={"x", "axis"},
    func_name="paddle.compat.min",
    correct_name="paddle.min",
)
def min(
    input: Tensor,
    *args: Any,
    out: Tensor | tuple[Tensor, Tensor] | list[Tensor] = None,
    **kwargs: Any,
) -> Tensor | MinMaxRetType:
    """

    Computes the minimum of tensor elements. There are mainly 3 cases (functionalities):

    1. paddle.compat.min(input: Tensor): reduce min over all dims, return a single value Tensor
    2. paddle.compat.min(input: Tensor, dim: int (cannot be None), keepdim=False): reduce min over the given dim,
        returns a named tuple MinMaxRetType(values: Tensor, indices: Tensor)
    3. paddle.compat.min(input: Tensor, other: Tensor): see `paddle.minimum`

    Special warning: the gradient behavior is NOT well-documented by PyTorch, the actual behavior should be:

    1. Case 1: the same as `min`
    2. Case 2: NOT evenly distributing the gradient for equal minimum elements! PyTorch actually only propagates to the elements with indices,
        for example: Tensor([1, 1, 1]) -> min(..., dim=0) -> values=Tensor(0, ...), indices=Tensor(0), the gradient for input tensor won't be
        Tensor([1/3, 1/3, 1/3]) as stated in their documentation, but will be Tensor([1, 0, 0]). This API implements a similar backward kernel.
    3. Case 3: the same as `minimum`

    Args:
        input (Tensor): A tensor, the data type is bfloat16, float16, float32, float64, int32, int64 on GPU.
            uint8, int32, int64, float32, float64 are allowed on CPU.
        dim (int, optional): The dim along which the minimum is computed.
            If this is not specified: see case 1, note that: `None` cannot be passed to this (TypeError will be thrown)
            compute the minimum over all elements of `input` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-input.ndim, input.ndim)`.
            If :math:`dim < 0`, the axis to reduce is :math:`input.ndim + dim`.
            Warning: if `dim` is specified, execute static graph will throw exceptions
            when not on a GPU device, since max_with_index is not implemented for non-GPU devices
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `input` unless :attr:`keepdim` is true, default
            value is False. Note that if `dim` does not appear in neither (`*args`) or (`**kwargs`), this parameter cannot be passed alone
        other (Tensor, optional): the other tensor to perform `paddle.minimum` with. This Tensor should
            have the same or broadcast-able shape as the `input`. Note that (`dim` & `keepdim`) and `other` are mutually exclusive
            meaning that trying to composite both will result in TypeError
        out (Tensor|tuple[Tensor, Tensor], optional): the output Tensor or tuple of (Tensor, int64 Tensor) that can be optionally
            given to be used as output buffers. For case 1 and 3 out is just a Tensor, while for case 2 we expect a tuple


    Returns:
        - For case 1. A single value Tensor (0-dim)
        - For case 2. A named tuple MinMaxRetType(values: Tensor, indices: Tensor), `values` has the same data type as the `input`,
            while indices is always an int64 Tensor, with exactly the same shape as `values`.
            MinMaxRetType can be used (indexed, packed, unpacked) in the same way as a regular tuple
        - For case 3. See `paddle.minimum` (:ref:`api_paddle_minimum`)


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # data_x is a Tensor with shape [2, 4]
            >>> # the axis is a int element
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]],
            ...                       dtype='float64', stop_gradient=False)
            >>> # Case 1: reduce over all dims
            >>> result1 = paddle.compat.min(x)
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=False,
            0.10000000)

            >>> # Case 2: reduce over specified dim
            >>> x.clear_grad()
            >>> result2 = paddle.compat.min(x, dim=1)
            >>> result2
            MinMaxRetType(values=Tensor(shape=[2], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                [0.20000000, 0.10000000]), indices=Tensor(shape=[2], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                [0, 0]))
            >>> result2[0].backward()
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                [[1., 0., 0., 0.],
                 [1., 0., 0., 0.]])

            >>> # Case 3: equivalent to `paddle.minimum`
            >>> x.clear_grad()
            >>> y = paddle.to_tensor([[0.5, 0.4, 0.1, 0.2],
            ...                       [0.3, 0.1, 0.6, 0.7]],
            ...                       dtype='float64', stop_gradient=False)
            >>> result3 = paddle.compat.min(x, y)
            >>> result3
            Tensor(shape=[2, 4], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                [[0.20000000, 0.30000000, 0.10000000, 0.20000000],
                 [0.10000000, 0.10000000, 0.60000000, 0.70000000]])
    """
    if not isinstance(input, (paddle.pir.Value, paddle.Tensor)):
        raise TypeError(
            f"input should be a tensor, but got an instance with type '{type(input).__name__}'"
        )
    _min_max_tensor_allow_grad(input)

    dim_or_other, keepdim = _min_max_param_checker("min", *args, **kwargs)

    ret = None
    if dim_or_other is None:
        # paddle.min and paddle.amin actually shares the same grad op (ReduceAminKernel)
        _check_out_status(out, False)
        ret = paddle.min(input)
    elif isinstance(dim_or_other, int):
        _check_out_status(out, True)
        if input.ndim:
            if in_dynamic_mode() and not input.place.is_gpu_place():
                _min_max_allow_cpu_composite(input)
                # CPUPlace and other placements are implemented by composition

                indices = paddle.argmin(input, axis=dim_or_other, keepdim=True)
                values = paddle.take_along_axis(
                    input, indices, axis=dim_or_other
                )
                if keepdim:
                    ret = MinMaxRetType(values=values, indices=indices)
                else:
                    ret = MinMaxRetType(
                        values=values.squeeze_(axis=dim_or_other),
                        indices=indices.squeeze_(axis=dim_or_other),
                    )
            else:
                vals, inds = _C_ops.min_with_index(
                    input, dim_or_other, keepdim, False
                )
                inds.stop_gradient = True
                ret = MinMaxRetType(values=vals, indices=inds)
        else:
            ret = MinMaxRetType(
                values=input,
                indices=paddle.zeros(
                    [], dtype=paddle.int64, device=input.place
                ),
            )
    else:
        _check_out_status(out, False)
        ret = _C_ops.minimum(input, dim_or_other)

    if out is not None:
        if isinstance(ret, MinMaxRetType):
            paddle.assign(ret.values, out[0])
            paddle.assign(ret.indices, out[1])
        else:
            paddle.assign(ret, out)
    return ret


@ForbidKeywordsDecorator(
    illegal_keys={"x", "axis"},
    func_name="paddle.compat.max",
    correct_name="paddle.max",
)
def max(
    input: Tensor,
    *args: Any,
    out: Tensor | tuple[Tensor, Tensor] | list[Tensor] = None,
    **kwargs: Any,
) -> Tensor | MinMaxRetType:
    """

    Computes the maximum of tensor elements. There are mainly 3 cases (functionalities):

    1. paddle.compat.max(input: Tensor): reduce max over all dims, return a single value Tensor
    2. paddle.compat.max(input: Tensor, dim: int (cannot be None), keepdim=False): reduce max over the given dim,
        returns a named tuple MinMaxRetType(values: Tensor, indices: Tensor)
    3. paddle.compat.max(input: Tensor, other: Tensor): see `paddle.maximum`

    Special warning: the gradient behavior is NOT well-documented by PyTorch, the actual behavior should be:

    1. Case 1: the same as `max`
    2. Case 2: NOT evenly distributing the gradient for equal maximum elements! PyTorch actually only propagates to the elements with indices,
        for example: Tensor([1, 1, 1]) -> max(..., dim=0) -> values=Tensor(0, ...), indices=Tensor(0), the gradient for input tensor won't be
        Tensor([1/3, 1/3, 1/3]) as stated in their documentation, but will be Tensor([1, 0, 0]). This API implements a similar backward kernel.
    3. Case 3: the same as `maximum`

    Args:
        input (Tensor): A tensor, the data type is bfloat16, float16, float32, float64, int32, int64 on GPU.
            uint8, int32, int64, float32, float64 are allowed on CPU.
        dim (int, optional): The dim along which the maximum is computed.
            If this is not specified: see case 1, note that: `None` cannot be passed to this (TypeError will be thrown)
            compute the maximum over all elements of `input` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-input.ndim, input.ndim)`.
            If :math:`dim < 0`, the axis to reduce is :math:`input.ndim + dim`.
            Warning: if `dim` is specified, execute static graph will throw exceptions
            when not on a GPU device, since max_with_index is not implemented for non-GPU devices
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `input` unless :attr:`keepdim` is true, default
            value is False. Note that if `dim` does not appear in neither (`*args`) or (`**kwargs`), this parameter cannot be passed alone
        other (Tensor, optional): the other tensor to perform `paddle.maximum` with. This Tensor should
            have the same or broadcast-able shape as the `input`. Note that (`dim` & `keepdim`) and `other` are mutually exclusive
            meaning that trying to composite both will result in TypeError
        out (Tensor|tuple[Tensor, Tensor], optional): the output Tensor or tuple of (Tensor, int64 Tensor) that can be optionally
            given to be used as output buffers. For case 1 and 3 out is just a Tensor, while for case 2 we expect a tuple


    Returns:
        - For case 1. A single value Tensor (0-dim)
        - For case 2. A named tuple MinMaxRetType(values: Tensor, indices: Tensor), `values` has the same data type as the `input`,
            while indices is always an int64 Tensor, with exactly the same shape as `values`.
            MinMaxRetType can be used (indexed, packed, unpacked) in the same way as a regular tuple
        - For case 3. See `paddle.maximum` (:ref:`api_paddle_maximum`)


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # data_x is a Tensor with shape [2, 4]
            >>> # the axis is a int element
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]],
            ...                       dtype='float64', stop_gradient=False)
            >>> # Case 1: reduce over all dims
            >>> result1 = paddle.compat.max(x)
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=False,
            0.90000000)

            >>> # Case 2: reduce over specified dim
            >>> x.clear_grad()
            >>> result2 = paddle.compat.max(x, dim=1)
            >>> result2
            MinMaxRetType(values=Tensor(shape=[2], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                [0.90000000, 0.70000000]), indices=Tensor(shape=[2], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                [3, 3]))
            >>> result2[0].backward()
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                [[0., 0., 0., 1.],
                 [0., 0., 0., 1.]])

            >>> # Case 3: equivalent to `paddle.maximum`
            >>> x.clear_grad()
            >>> y = paddle.to_tensor([[0.5, 0.4, 0.1, 0.2],
            ...                       [0.3, 0.1, 0.6, 0.7]],
            ...                       dtype='float64', stop_gradient=False)
            >>> result3 = paddle.compat.max(x, y)
            >>> result3
            Tensor(shape=[2, 4], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                [[0.50000000, 0.40000000, 0.50000000, 0.90000000],
                 [0.30000000, 0.20000000, 0.60000000, 0.70000000]])
    """
    if not isinstance(input, (paddle.pir.Value, paddle.Tensor)):
        raise TypeError(
            f"input should be a tensor, but got an instance with type '{type(input).__name__}'"
        )
    _min_max_tensor_allow_grad(input)

    dim_or_other, keepdim = _min_max_param_checker("max", *args, **kwargs)

    ret = None
    if dim_or_other is None:
        _check_out_status(out, False)
        ret = paddle.max(input)
    elif isinstance(dim_or_other, int):
        _check_out_status(out, True)
        if input.ndim:
            if in_dynamic_mode() and not input.place.is_gpu_place():
                _min_max_allow_cpu_composite(input)
                indices = paddle.argmax(input, axis=dim_or_other, keepdim=True)
                values = paddle.take_along_axis(
                    input, indices, axis=dim_or_other
                )
                if keepdim:
                    ret = MinMaxRetType(values=values, indices=indices)
                else:
                    ret = MinMaxRetType(
                        values=values.squeeze_(axis=dim_or_other),
                        indices=indices.squeeze_(axis=dim_or_other),
                    )
            else:
                vals, inds = _C_ops.max_with_index(
                    input, dim_or_other, keepdim, False
                )
                inds.stop_gradient = True
                ret = MinMaxRetType(values=vals, indices=inds)
        else:
            ret = MinMaxRetType(
                values=input,
                indices=paddle.zeros(
                    [], dtype=paddle.int64, device=input.place
                ),
            )
    else:
        _check_out_status(out, False)
        ret = _C_ops.maximum(input, dim_or_other)

    if out is not None:
        if isinstance(ret, MinMaxRetType):
            paddle.assign(ret.values, out[0])
            paddle.assign(ret.indices, out[1])
        else:
            paddle.assign(ret, out)
    return ret


class SlogdetResult(NamedTuple):
    sign: Tensor
    logabsdet: Tensor


def slogdet(x: Tensor, out: SlogdetResult | None = None) -> SlogdetResult:
    """
    (PyTorch Compatible API) Calculates the sign and natural logarithm of the absolute value of a square matrix's or batches square matrices' determinant.
    The determinant can be computed with ``sign * exp`` (logabsdet).

    Supports input of float, double, complex64, complex128.

    Notes:
        1. For matrices that have zero determinant, this returns ``(0, -inf)``.

        2. For matrices with complex value, the :math:`abs(det)` is the modulus of the determinant,
        and therefore :math:`sign = det / abs(det)`.

        3. The return structure of this API has been revised **from a single stacked Tensor of shape `[2, *]` (where index 0 was sign and index 1 was logabsdet) to a tuple of two independent Tensors `(sign, logabsdet)`** (see `PR #72505 <https://github.com/PaddlePaddle/Paddle/pull/72505>`_).
        This modification may cause incompatibility with models previously exported for inference that relied on the old return structure.

    Args:
        x (Tensor): the batch of matrices of size :math:`(*, n, n)`
            where math:`*` is one or more batch dimensions.
        out(SlogdetResult, optional): The tuple of output tensor, contains ``abs`` and ``logabsdet``.

    Returns:
        SlogdetResult: A tuple containing two Tensors: (sign, logabsdet).
        The first Tensor represents the signs of the determinants and the second Tensor
        represents the natural logarithms of the absolute values of the determinants.
        Each output Tensor has a shape of :math:`(*)`, where :math:`*` matches the
        batch dimensions of the input `x`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1., 0.], [0., 1.]])
            >>> A = paddle.compat.slogdet(x)
            >>> print(A.sign)
            Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   1.)
            >>> print(A.logabsdet)
            Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   0.)
    """
    sign, logabsdet = _C_ops.slogdet_v2(x, out=out)
    if out is not None:
        paddle.assign(sign, out[0])
        paddle.assign(logabsdet, out[1])
    return SlogdetResult(sign, logabsdet)


class SortRetType(NamedTuple):
    values: Tensor
    indices: Tensor


@ForbidKeywordsDecorator(
    illegal_keys={'x', 'axis'},
    func_name="paddle.compat.sort",
    correct_name='paddle.sort',
)
def sort(
    input: Tensor,
    dim: int = -1,
    descending: bool = False,
    stable: bool = False,
    out=None,
) -> SortRetType:
    """

    Sorts the input along the given dimension, and returns the sorted output and indices tensor. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.

    Args:
        input (Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8, float16, bfloat16
        dim (int, optional): Dimension to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when dim<0, it works the same way
            as dim+R. Default is -1.
        descending (bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        stable (bool, optional): Whether to use stable sorting algorithm or not.
            When using stable sorting algorithm, the order of equivalent elements
            will be preserved. Default is False.
        out (tuple, optional) : the output tuple/list of (Tensor, Tensor) that
            can be optionally given to be used as output buffers

    Returns:
        SortRetType, a named tuple which contains `values` and `indices`, can be accessed through either indexing
        (e.g. `result[0]` for values and `result[1]` for indices), or by `result.values` & `result.indices`

    Examples:

    .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[5,8,9,5],
            ...                       [0,0,1,7],
            ...                       [6,9,2,4]],
            ...                      dtype='float32')
            >>> out1 = paddle.compat.sort(input=x, dim=-1)
            >>> out2 = paddle.compat.sort(x, 1, descending=True)
            >>> out1
            SortRetType(values=Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[5., 5., 8., 9.],
                    [0., 0., 1., 7.],
                    [2., 4., 6., 9.]]), indices=Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
                   [[0, 3, 1, 2],
                    [0, 1, 2, 3],
                    [2, 3, 0, 1]]))
            >>> out2
            SortRetType(values=Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[9., 8., 5., 5.],
                    [7., 1., 0., 0.],
                    [9., 6., 4., 2.]]), indices=Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
                   [[2, 1, 0, 3],
                    [3, 2, 0, 1],
                    [1, 0, 3, 2]]))
    """
    _check_out_status(out, expect_multiple=True)
    outputs, indices = _C_ops.argsort(input, dim, descending, stable)
    if out is not None:
        paddle.assign(outputs, out[0])
        paddle.assign(indices, out[1])
    return SortRetType(values=outputs, indices=indices)


@overload
def unique(
    input: Tensor,
    sorted: bool = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def unique(
    input: Tensor,
    sorted: bool = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def unique(
    input: Tensor,
    sorted: bool = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def unique(
    input: Tensor,
    sorted: bool = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = ...,
) -> Tensor: ...


@ForbidKeywordsDecorator(
    illegal_keys={"x", "axis"},
    func_name="paddle.compat.unique",
    correct_name="paddle.unique",
)
def unique(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    r"""
    Returns the unique elements of `input` in ascending order.

    Args:
        input(Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
        sorted(bool, optional): Does not affect the return result, same as PyTorch.
        return_inverse(bool, optional): If True, also return the indices for where elements in
            the original input ended up in the returned unique tensor.
        return_counts(bool, optional): If True, also return the counts for each unique element.
        dim(int, optional): The axis to apply unique. If None, the input will be flattened.
            Default: None.

    Returns:
        tuple (output, inverse_indices, counts). `output` is the unique tensor for `input`. \
            `inverse_indices` is provided only if `return_inverse` \
            is True. `counts` is provided only if `return_counts` is True.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
            >>> unique = paddle.compat.unique(x)
            >>> print(unique)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 2, 3, 5])

            >>> _, inverse_indices, counts = paddle.compat.unique(x, return_inverse=True, return_counts=True)
            >>> print(inverse_indices)
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 2, 2, 0, 3, 2])
            >>> print(counts)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 1, 3, 1])

            >>> x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
            >>> unique = paddle.compat.unique(x)
            >>> print(unique)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 1, 2, 3])

            >>> unique = paddle.compat.unique(x, dim=0)
            >>> print(unique)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[2, 1, 3],
             [3, 0, 1]])
    """
    return paddle.unique(
        input,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=dim,
        sorted=sorted,
    )


@ForbidKeywordsDecorator(
    illegal_keys={"x", "num_or_sections", "axis", "name"},
    func_name="paddle.compat.split",
    correct_name="paddle.split",
)
def split(
    tensor: Tensor, split_size_or_sections: int | Sequence[int], dim: int = 0
) -> tuple[Tensor, ...]:
    """
    (PyTorch Compatible API) Split the input tensor into multiple sub-Tensors.

    Args:
        tensor (Tensor): A N-D Tensor. The data type is bool, bfloat16, float16, float32, float64, uint8, int8, int32 or int64.
        split_size_or_sections (int|list|tuple):
            If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible).
            Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.
            If split_size_or_sections is a list, then tensor will be split into len(split_size_or_sections) chunks with sizes
            in dim according to split_size_or_sections. Negative inputs are not allowed. For example: for a dim with 9 channels,
            [2, 3, -1] will not be interpreted as [2, 3, 4], but will be rejected and an exception will be thrown.
        dim (int|Tensor, optional): The dim along which to split, it can be a integer or a ``0-D Tensor``
            with shape [] and data type  ``int32`` or ``int64``.
            If :math::`dim < 0`, the dim to split along is :math:`rank(x) + dim`. Default is 0.
    Returns:
        tuple(Tensor), The tuple of segmented Tensors.

    Note:
        This is a pytorch compatible API that follows the function signature and behavior of torch.split.
        To use the original split of paddle, please consider `paddle.split`

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> # x is a Tensor of shape [3, 8, 5]
            >>> x = paddle.rand([3, 8, 5])

            >>> out0, out1, out2 = paddle.compat.split(x, split_size_or_sections=3, dim=1)
            >>> print(out0.shape)
            paddle.Size([3, 3, 5])
            >>> print(out1.shape)
            paddle.Size([3, 3, 5])
            >>> print(out2.shape)
            paddle.Size([3, 2, 5])

            >>> out0, out1, out2 = paddle.compat.split(x, split_size_or_sections=[1, 2, 5], dim=1)
            >>> print(out0.shape)
            paddle.Size([3, 1, 5])
            >>> print(out1.shape)
            paddle.Size([3, 2, 5])
            >>> print(out2.shape)
            paddle.Size([3, 5, 5])

            >>> # dim is negative, the real dim is (rank(x) + dim)=1
            >>> out0, out1, out2 = paddle.compat.split(x, split_size_or_sections=3, dim=-2)
            >>> print(out0.shape)
            paddle.Size([3, 3, 5])
            >>> print(out1.shape)
            paddle.Size([3, 3, 5])
            >>> print(out2.shape)
            paddle.Size([3, 2, 5])
    """

    def GetSplitSize(split_size, shape_on_dim):
        remaining_num = shape_on_dim % split_size_or_sections
        num_complete_section = shape_on_dim // split_size_or_sections
        if remaining_num == 0:
            return num_complete_section
        else:
            sections = [
                split_size_or_sections for _ in range(num_complete_section)
            ]
            sections.append(remaining_num)
            return sections

    def GetShapeOnDimInRange(shape, dim: int) -> int:
        shape_range = len(shape)
        if isinstance(dim, int):
            if dim < -shape_range or dim >= shape_range:
                raise ValueError(
                    f"(InvalidArgument) The dim is expected to be in range of [-{shape_range}, {shape_range}), but got {dim}"
                )
        return shape[dim]

    if isinstance(split_size_or_sections, (list, tuple)):
        for i, section_size in enumerate(split_size_or_sections):
            shape_val = 0
            if isinstance(section_size, Variable):
                shape_val = int(section_size.item(0))
            else:
                shape_val = section_size
            if section_size < 0:
                raise ValueError(
                    f"paddle.compat.split expects split_sizes have only non-negative entries, but got size = {section_size} on dim {i}"
                )

    if in_dynamic_mode():
        if isinstance(dim, Variable):
            dim = dim.item(0)
        assert dim + len(tensor.shape) >= 0, "(rank(x) + dim) must >= 0"
        dim = (dim + len(tensor.shape)) if dim < 0 else dim

        if isinstance(split_size_or_sections, (list, tuple)):
            if paddle.utils._contain_var(split_size_or_sections):
                for index, item in enumerate(split_size_or_sections):
                    if isinstance(item, Variable):
                        split_size_or_sections[index] = split_size_or_sections[
                            index
                        ].item()
        elif not isinstance(split_size_or_sections, int):
            raise TypeError(
                "The type of 'split_size_or_sections' in split must be int, list or tuple in imperative mode, but "
                f"received {type(split_size_or_sections)}."
            )

        if isinstance(split_size_or_sections, int):
            # check whether shape is divisible
            assert split_size_or_sections > 0, (
                'split_size_or_sections must be greater than 0.'
            )

            split_size_or_sections = GetSplitSize(
                split_size_or_sections, GetShapeOnDimInRange(tensor.shape, dim)
            )

            if isinstance(split_size_or_sections, list):
                return tuple(_C_ops.split(tensor, split_size_or_sections, dim))
            else:
                return tuple(
                    _C_ops.split_with_num(tensor, split_size_or_sections, dim)
                )
        else:
            return tuple(_C_ops.split(tensor, split_size_or_sections, dim))
    else:
        if isinstance(dim, paddle.pir.Value):
            raise TypeError(
                "'dim' is not allowed to be a pir.Value in a static graph: "
                "\npir.Value can not be used for indexing python lists/tuples."
            )
        if isinstance(dim, int):
            assert len(tensor.shape) + dim >= 0, "(rank(x) + dim) must >= 0"
            dim = (len(tensor.shape) + dim) if dim < 0 else dim

        input_shape = tensor.shape

        if not isinstance(split_size_or_sections, (int, list, tuple)):
            raise TypeError(
                "The type of 'split_size_or_sections' in split must be int, list or tuple in imperative mode."
            )
        if isinstance(split_size_or_sections, int):
            assert split_size_or_sections > 0, (
                'split_size_or_sections must be greater than 0.'
            )

            split_size_or_sections = GetSplitSize(
                split_size_or_sections, GetShapeOnDimInRange(tensor.shape, dim)
            )
            if isinstance(split_size_or_sections, list):
                if paddle.utils._contain_var(split_size_or_sections):
                    split_size_or_sections = paddle.utils.get_int_tensor_list(
                        split_size_or_sections
                    )
                return tuple(_C_ops.split(tensor, split_size_or_sections, dim))
            else:
                return tuple(
                    _C_ops.split_with_num(tensor, split_size_or_sections, dim)
                )
        else:
            if isinstance(dim, int) and input_shape[dim] > 0:
                assert len(split_size_or_sections) <= input_shape[dim], (
                    'len(split_size_or_sections) must not be more than input.shape[dim].'
                )
            if paddle.utils._contain_var(split_size_or_sections):
                split_size_or_sections = paddle.utils.get_int_tensor_list(
                    split_size_or_sections
                )
            return tuple(_C_ops.split(tensor, split_size_or_sections, dim))
