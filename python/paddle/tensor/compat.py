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

from typing import TYPE_CHECKING, NamedTuple

import paddle
from paddle import _C_ops

from ..base.framework import Variable
from ..framework import (
    in_dynamic_mode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle._typing import (
        Size2,
    )


from paddle import nn
from paddle.utils.decorator_utils import ForbidKeywordsDecorator

__all__ = []


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


class SortRetType(NamedTuple):
    values: Tensor
    indices: Tensor


def _check_out_status(
    out: Tensor | tuple[Tensor, Tensor] | list[Tensor],
    expect_multiple: bool = False,
):
    if out is None:
        return
    if not in_dynamic_mode():
        raise RuntimeError(
            "Using `out` static graph CINN backend is currently not supported. Directly return the tensor tuple instead.\n"
        )
    if expect_multiple:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise TypeError(
                f"Expected a list or tuple of two tensors, got {type(out)} instead."
            )
        if not (
            isinstance(out[0], paddle.Tensor)
            and isinstance(out[1], paddle.Tensor)
        ):
            raise TypeError(
                f"Expected Tensor type in the tuple/list, got ({type(out[0])}, {type(out[1])}) instead."
            )
    else:
        if not isinstance(out, paddle.Tensor):
            raise TypeError(f"Expected a Tensor, got {type(out)} instead.")


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
    if out is None:
        return SortRetType(values=outputs, indices=indices)
    paddle.assign(outputs, out[0])
    paddle.assign(indices, out[1])


class Unfold(nn.Unfold):
    """
    A compatible version of paddle.nn.Unfold:
    - The keyword arguments are in non-plural forms, example: `kernel_size` instead of kernel_sizes
    - `padding` restricts the size of the input to be 1(int) or 2, Size4 is not allowed. To use a more
       input-flexible version of Unfold, please refer to `paddle.nn.Unfold`.
    - All the input parameters allow `Tensor` or `pir.Value` as inputs, and will be converted to list
    Other aspects are the same. See ``paddle.nn.Unfold`` for more details.
    Parameters:
        kernel_size(int|list|tuple|Tensor): The size of convolution kernel, should be [k_h, k_w]
            or an integer k treated as [k, k].
        stride(int|list|tuple|Tensor, optional): The strides, should be [stride_h, stride_w]
            or an integer stride treated as [sride, stride]. For default, strides will be [1, 1].
        padding(int|list|tuple|Tensor, optional): The paddings of each dimension, should be
            a single integer or [padding_h, padding_w]. If [padding_h, padding_w] was given, it will expanded to
            [padding_h, padding_w, padding_h, padding_w]. If an integer padding was given,
            [padding, padding, padding, padding] will be used. By default, paddings will be 0.
        dilation(int|list|tuple|Tensor, optional): The dilations of convolution kernel, should be
            [dilation_h, dilation_w], or an integer dilation treated as [dilation, dilation].
            For default, it will be [1, 1].
    Examples:
        .. code-block:: python
            >>> import paddle
            >>> x = paddle.randn((100, 3, 224, 224))
            >>> unfold = paddle.compat.Unfold(kernel_size=[3, 3])
            >>> result = unfold(x)
            >>> print(result.shape)
            [100, 27, 49284]
    """

    kernel_sizes: Size2
    dilations: Size2
    paddings: Size2
    strides: Size2

    @ForbidKeywordsDecorator(
        illegal_keys={"kernel_sizes", "dilations", "paddings", "strides"},
        func_name="paddle.compat.Unfold",
        correct_name="paddle.nn.Unfold",
    )
    def __init__(
        self,
        kernel_size: Size2,
        dilation: Size2 = 1,
        padding: Size2 = 0,
        stride: Size2 = 1,
    ) -> None:
        super().__init__(kernel_size, dilation, padding, stride)

    def forward(self, input: Tensor) -> Tensor:
        def to_list_if_necessary(x, size_check=False):
            res = x
            if in_dynamic_mode() and isinstance(
                x, (paddle.pir.Value, paddle.Tensor)
            ):
                res = x.tolist()
            else:
                if not isinstance(x, (list, tuple, int)):
                    raise TypeError(
                        "paddle.compat.Unfold does not allow paddle.Tensor or pir.Value as inputs in static graph mode."
                    )
            if size_check and isinstance(res, (list, tuple)) and len(res) > 2:
                raise ValueError(
                    f"The `padding` field of paddle.compat.Unfold can only have size 1 or 2, now len={len(res)}. \nDid you mean to use paddle.nn.Unfold() instead?"
                )
            return res

        return nn.functional.unfold(
            input,
            kernel_sizes=to_list_if_necessary(self.kernel_sizes),
            strides=to_list_if_necessary(self.strides),
            paddings=to_list_if_necessary(self.paddings, size_check=True),
            dilations=to_list_if_necessary(self.dilations),
            name=self.name,
        )
