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

from typing import TYPE_CHECKING, Any, NamedTuple

import paddle
from paddle import _C_ops

from ..base.framework import Variable
from ..framework import (
    in_dynamic_mode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor

from paddle.utils.decorator_utils import ForbidKeywordsDecorator

__all__ = []


@ForbidKeywordsDecorator(
    illegal_keys=["x", "num_or_sections", "axis", "name"],
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
        .. code-block:: python

            >>> import paddle

            >>> # x is a Tensor of shape [3, 8, 5]
            >>> x = paddle.rand([3, 8, 5])

            >>> out0, out1, out2 = paddle.compat.split(x, split_size_or_sections=3, dim=1)
            >>> print(out0.shape)
            [3, 3, 5]
            >>> print(out1.shape)
            [3, 3, 5]
            >>> print(out2.shape)
            [3, 2, 5]

            >>> out0, out1, out2 = paddle.compat.split(x, split_size_or_sections=[1, 2, 5], dim=1)
            >>> print(out0.shape)
            [3, 1, 5]
            >>> print(out1.shape)
            [3, 2, 5]
            >>> print(out2.shape)
            [3, 5, 5]

            >>> # dim is negative, the real dim is (rank(x) + dim)=1
            >>> out0, out1, out2 = paddle.compat.split(x, split_size_or_sections=3, dim=-2)
            >>> print(out0.shape)
            [3, 3, 5]
            >>> print(out1.shape)
            [3, 3, 5]
            >>> print(out2.shape)
            [3, 2, 5]
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
            assert (
                split_size_or_sections > 0
            ), 'split_size_or_sections must be greater than 0.'

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
            assert (
                split_size_or_sections > 0
            ), 'split_size_or_sections must be greater than 0.'

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
                assert (
                    len(split_size_or_sections) <= input_shape[dim]
                ), 'len(split_size_or_sections) must not be more than input.shape[dim].'
            if paddle.utils._contain_var(split_size_or_sections):
                split_size_or_sections = paddle.utils.get_int_tensor_list(
                    split_size_or_sections
                )
            return tuple(_C_ops.split(tensor, split_size_or_sections, dim))


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
        found_key = None

    dim_or_other = None
    keepdim = False

    num_args = len(args)
    total_arg_num = num_args + len(kwargs)
    if total_arg_num > 2:
        raise invalid_arguments_exception()
    elif total_arg_num == 2:
        if num_args == 2:
            dim_or_other, keepdim = args
            if dim_or_other is None or isinstance(
                dim_or_other, (Variable, paddle.pir.Value)
            ):
                raise invalid_arguments_exception()
        elif num_args == 1:
            dim_or_other = args[0]
            if dim_or_other is None or isinstance(
                dim_or_other, (Variable, paddle.pir.Value)
            ):
                raise invalid_arguments_exception()
            keepdim = try_get_keys("keepdim")
        else:
            dim_or_other = try_get_keys("dim")
            keepdim = try_get_keys("keepdim")
    elif total_arg_num == 1:
        if num_args:
            dim_or_other = args[0]
            if dim_or_other is None:
                raise invalid_arguments_exception()
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
            f"The second input must be int or Tensor or implicit None in compat.min, but received {type(dim_or_other)}.\n"
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
        if not input.place.is_gpu_place():
            raise TypeError(
                f"Non-CUDA GPU placed Tensor does not have '{in_dtype}' op registered.\n"
                "Paddle support following DataTypes: int32, int64, float64, float32, uint8"
            )


@ForbidKeywordsDecorator(
    illegal_keys=['x', 'axis'],
    func_name="paddle.compat.min",
    correct_name='paddle.min',
)
def min(input: Tensor, *args: Any, **kwargs: Any) -> Tensor | MinMaxRetType:
    """

    Computes the minimum of tensor elements. There are mainly 3 cases (functionalities):
    1. paddle.compat.min(input: Tensor): reduce min over all dims, return a single value Tensor
    2. paddle.compat.min(input: Tensor, dim: int (cannot be None), keepdim=False): reduce min over the given dim,
        returns a named tuple MinMaxRetType(values: Tensor, indices: Tensor)
    3. paddle.compat.min(input: Tensor, other: Tensor): see `paddle.minimum`

    Note: If there are multiple minimum elements, this API evenly distributes gradient between these equal values,
        following torch.min. The gradient behavior of `values` for case 2 is the same as `paddle.amin`.

    Args:
        input (Tensor): A tensor, the data type is bfloat16, float16, float32, float64, int32, int64 on GPU.
            uint8, int32, int64, float32, float64 are allowed on CPU.
        dim (int, optional): The dim along which the minimum is computed.
            If this is not specified: see case 1, note that: `None` cannot be passed to this (TypeError will be thrown)
            compute the minimum over all elements of `input` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-input.ndim, input.ndim)`.
            If :math:`dim < 0`, the axis to reduce is :math:`input.ndim + dim`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `input` unless :attr:`keepdim` is true, default
            value is False. Note that if `dim` does not appear in neither (*args) or (**kwargs), this parameter cannot be passed alone
        other (Tensor, optional): the other tensor to perform `paddle.minimum` with. This Tensor should
            have the same or broadcast-able shape as the `input`. Note that (`dim` & `keepdim`) and `other` are mutually exclusive
            meaning that trying to composite both will result in TypeError

    Returns:
        - For case 1: a single value Tensor (0-dim)
        - For case 2: a named tuple MinMaxRetType(values: Tensor, indices: Tensor), `values` has the same data type as the `input`,
            while indices is always an int64 Tensor, with exactly the same shape as `values`.
            MinMaxRetType can be used (indexed, packed, unpacked) in the same way as a regular tuple
        - For case 3: see `paddle.minimum`


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
    if not isinstance(input, paddle.pir.Value) and not isinstance(
        input, paddle.Tensor
    ):
        raise TypeError(
            f"input should be a tensor, but got an instance with type '{type(input).__name__}'"
        )
    _min_max_tensor_allow_grad(input)

    dim_or_other, keepdim = _min_max_param_checker("min", *args, **kwargs)

    if dim_or_other is None:
        return _C_ops.min(input, None, False)
    elif isinstance(dim_or_other, int):
        if input.place.is_gpu_place():
            vals, inds = _C_ops.min_with_index(
                input, dim_or_other, keepdim, False
            )
            inds.stop_gradient = True
            return MinMaxRetType(values=vals, indices=inds)
        else:
            _min_max_allow_cpu_composite(input)
            # CPUPlace and other placements are implemented by composition
            indices = paddle.argmin(input, axis=dim_or_other, keepdim=True)
            values = paddle.take_along_axis(input, indices, axis=dim_or_other)
            if keepdim:
                return MinMaxRetType(values=values, indices=indices)
            return MinMaxRetType(
                values=values.squeeze_(axis=dim_or_other),
                indices=indices.squeeze_(axis=dim_or_other),
            )
    else:
        return _C_ops.minimum(input, dim_or_other)


@ForbidKeywordsDecorator(
    illegal_keys=['x', 'axis'],
    func_name="paddle.compat.max",
    correct_name='paddle.max',
)
def max(input: Tensor, *args: Any, **kwargs: Any) -> Tensor | MinMaxRetType:
    """

    Computes the maximum of tensor elements. There are mainly 3 cases (functionalities):
    1. paddle.compat.max(input: Tensor): reduce max over all dims, return a single value Tensor
    2. paddle.compat.max(input: Tensor, dim: int (cannot be None), keepdim=False): reduce max over the given dim,
        returns a named tuple MinMaxRetType(values: Tensor, indices: Tensor)
    3. paddle.compat.max(input: Tensor, other: Tensor): see `paddle.maximum`

    Note: If there are multiple maximum elements, this API evenly distributes gradient between these equal values,
        following torch.max. The gradient behavior of `values` for case 2 is the same as `paddle.amax`.

    Args:
        input (Tensor): A tensor, the data type is bfloat16, float16, float32, float64, int32, int64 on GPU.
            uint8, int32, int64, float32, float64 are allowed on CPU.
        dim (int, optional): The dim along which the maximum is computed.
            If this is not specified: see case 1, note that: `None` cannot be passed to this (TypeError will be thrown)
            compute the maximum over all elements of `input` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-input.ndim, input.ndim)`.
            If :math:`dim < 0`, the axis to reduce is :math:`input.ndim + dim`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `input` unless :attr:`keepdim` is true, default
            value is False. Note that if `dim` does not appear in neither (*args) or (**kwargs), this parameter cannot be passed alone
        other (Tensor, optional): the other tensor to perform `paddle.maximum` with. This Tensor should
            have the same or broadcast-able shape as the `input`. Note that (`dim` & `keepdim`) and `other` are mutually exclusive
            meaning that trying to composite both will result in TypeError

    Returns:
        - For case 1: a single value Tensor (0-dim)
        - For case 2: a named tuple MinMaxRetType(values: Tensor, indices: Tensor), `values` has the same data type as the `input`,
            while indices is always an int64 Tensor, with exactly the same shape as `values`.
            MinMaxRetType can be used (indexed, packed, unpacked) in the same way as a regular tuple
        - For case 3: see `paddle.maximum`


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
    if not isinstance(input, paddle.pir.Value) and not isinstance(
        input, paddle.Tensor
    ):
        raise TypeError(
            f"input should be a tensor, but got an instance with type '{type(input).__name__}'"
        )
    _min_max_tensor_allow_grad(input)

    dim_or_other, keepdim = _min_max_param_checker("max", *args, **kwargs)

    if dim_or_other is None:
        return _C_ops.max(input, None, False)
    elif isinstance(dim_or_other, int):
        if input.place.is_gpu_place():
            vals, inds = _C_ops.max_with_index(
                input, dim_or_other, keepdim, False
            )
            inds.stop_gradient = True
            return MinMaxRetType(values=vals, indices=inds)
        else:
            _min_max_allow_cpu_composite(input)
            # CPUPlace and other placements are implemented by composition
            indices = paddle.argmax(input, axis=dim_or_other, keepdim=True)
            values = paddle.take_along_axis(input, indices, axis=dim_or_other)
            if keepdim:
                return MinMaxRetType(values=values, indices=indices)
            return MinMaxRetType(
                values=values.squeeze_(axis=dim_or_other),
                indices=indices.squeeze_(axis=dim_or_other),
            )
    else:
        return _C_ops.maximum(input, dim_or_other)
