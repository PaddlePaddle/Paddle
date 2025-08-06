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

import paddle
from paddle import _C_ops
from paddle.tensor import fill_constant

from ..base.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
)
from ..base.framework import Variable
from ..framework import (
    LayerHelper,
    in_dynamic_mode,
    in_pir_mode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor

from paddle.utils.compat_kwarg_check import forbid_keywords

__all__ = []


@forbid_keywords(["x", "num_or_sections", "axis", "name"], "paddle.split")
def split(
    tensor: Tensor, split_size_or_sections: int | Sequence[int], dim: int = 0
) -> tuple[Tensor]:
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

            >>> # x is a Tensor of shape [3, 9, 5]
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

    def SaveGetShapeOnDim(shape, dim: int) -> int:
        shape_range = len(shape)
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
                split_size_or_sections, SaveGetShapeOnDim(tensor.shape, dim)
            )

            if isinstance(split_size_or_sections, list):
                return tuple(_C_ops.split(tensor, split_size_or_sections, dim))
            else:
                return tuple(
                    _C_ops.split_with_num(tensor, split_size_or_sections, dim)
                )
        else:
            return tuple(_C_ops.split(tensor, split_size_or_sections, dim))
    elif in_pir_mode():
        if isinstance(dim, paddle.pir.Value):
            dim.stop_gradient = True
        if isinstance(dim, int):
            assert len(tensor.shape) + dim >= 0, "(rank(x) + dim) must >= 0"
            dim = (len(tensor.shape) + dim) if dim < 0 else dim

        input_shape = tensor.shape

        if not isinstance(split_size_or_sections, (int, list, tuple)):
            raise TypeError(
                "The type of 'split_size_or_sections' in split must be int, list or tuple in imperative mode, but "
                f"received {type(split_size_or_sections)}."
            )
        if isinstance(split_size_or_sections, int):
            assert (
                split_size_or_sections > 0
            ), 'split_size_or_sections must be greater than 0.'

            split_size_or_sections = GetSplitSize(
                split_size_or_sections, SaveGetShapeOnDim(tensor.shape, dim)
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

    else:
        check_variable_and_dtype(
            tensor,
            'input',
            [
                'bool',
                'bfloat16',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint8',
                'int8',
            ],
            'split',
        )
        check_type(
            split_size_or_sections,
            'split_size_or_sections',
            (list, int, tuple),
            'split',
        )
        check_type(dim, 'dim', (int, Variable), 'split')
        if isinstance(dim, Variable):
            check_dtype(dim.dtype, 'dim', ['int32', 'int64'], 'split')

        helper = LayerHelper('split', **locals())

        input_shape = tensor.shape
        inputs = {'X': tensor}
        attrs = {'num': 0}

        def _get_SectionsTensorList(one_list):
            tensor_list = []
            unk_dim_idx = -1
            for idx, dim_size in enumerate(one_list):
                if isinstance(dim_size, Variable):
                    dim_size.stop_gradient = True
                    tensor_list.append(dim_size)
                else:
                    assert isinstance(dim_size, int)
                    if dim_size == -1:
                        assert unk_dim_idx == -1, (
                            "Only one value of 'num_or_section' in split can "
                            f"be -1. But received num_or_section[{idx}] is also -1."
                        )
                        unk_dim_idx = idx
                    temp_out = helper.create_variable_for_type_inference(
                        'int32'
                    )
                    fill_constant(
                        [1], 'int32', dim_size, force_cpu=True, out=temp_out
                    )
                    tensor_list.append(temp_out)
            return tuple(tensor_list)

        if isinstance(dim, Variable):
            dim.stop_gradient = True
            inputs['AxisTensor'] = dim
        else:
            assert len(tensor.shape) + dim >= 0, "(rank(x) + dim) must >= 0"
            dim = (len(input_shape) + dim) if dim < 0 else dim
            attrs['axis'] = dim

        if isinstance(split_size_or_sections, int):
            shape_on_dim = SaveGetShapeOnDim(tensor.shape, dim)
            split_size_or_sections = GetSplitSize(
                split_size_or_sections, shape_on_dim
            )

        if isinstance(split_size_or_sections, int):
            # after GetSplitSize, if the result is int, split_size_or_sections is actually equivalent to the original num_or_sections (num)
            attrs['num'] = split_size_or_sections
            assert (
                split_size_or_sections > 0
            ), 'split_size_or_sections must be than 0.'
            num = split_size_or_sections
        else:
            if isinstance(dim, int) and input_shape[dim] > 0:
                assert (
                    len(split_size_or_sections) <= input_shape[dim]
                ), 'len(split_size_or_sections) must not be more than input.shape[dim].'
            num = len(split_size_or_sections)
            attrs['sections'] = [
                -1 if isinstance(ele, Variable) else ele
                for ele in split_size_or_sections
            ]
            if paddle.utils._contain_var(split_size_or_sections):
                inputs['SectionsTensorList'] = _get_SectionsTensorList(
                    split_size_or_sections
                )

        outs = [
            helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()
            )
            for i in range(num)
        ]
        helper.append_op(
            type='split', inputs=inputs, outputs={'Out': outs}, attrs=attrs
        )
        return tuple(outs)
