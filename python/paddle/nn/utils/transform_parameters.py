# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from functools import reduce
from typing import TYPE_CHECKING

import paddle
from paddle import _C_ops
from paddle.base.framework import (
    _create_tensor,
    _dygraph_tracer,
    dygraph_only,
    in_dygraph_mode,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from paddle import Tensor
    from paddle._typing import ShapeLike


# input==output, inplace strategy of reshape has no cost almost
def _inplace_reshape_dygraph(x: Tensor, shape: ShapeLike) -> None:
    x_shape = _create_tensor(dtype='int64')
    if in_dygraph_mode():
        with paddle.base.dygraph.no_grad():
            tmp_out = _C_ops.reshape(x, shape)
            tmp_out._share_underline_tensor_to(x)
    else:
        _dygraph_tracer().trace_op(
            type="reshape2",
            inputs={'X': x},
            outputs={'Out': x, 'XShape': x_shape},
            attrs={'shape': shape},
            stop_gradient=True,
        )


@dygraph_only
def _stride_column(param: Tensor) -> None:
    """
    A tool function. Permute date of parameter as a 'columns' stride. Now, it only support 2-D parameter.

    Args:
        param(Tensor): The param that will be strided according to 'columns'.

    Examples:
       .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> linear = paddle.nn.Linear(2, 3)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [[ 0.11732829, -0.64161885, -1.06996548],
                    [ 0.03456247, -0.29862350, -0.52380574]])

            >>> paddle.nn.utils._stride_column(linear.weight)
            >>> print(linear.weight)

    """
    assert len(param.shape) == 2
    shape = [param.shape[1], param.shape[0]]
    with paddle.base.dygraph.no_grad():
        reshape_var = paddle.reshape(param, shape)
        transpose_var = paddle.transpose(reshape_var, [1, 0])
        transpose_var._share_underline_tensor_to(param)


@dygraph_only
def parameters_to_vector(
    parameters: Iterable[Tensor], name: str | None = None
) -> Tensor:
    """
    Flatten parameters to a 1-D Tensor.

    Args:
        parameters(Iterable[Tensor]): Iterable Tensors that are trainable parameters of a Layer.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A 1-D Tensor, which represents the parameters of a Layer.


    Examples:
       .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> linear = paddle.nn.Linear(10, 15)

            >>> t = paddle.nn.utils.parameters_to_vector(linear.parameters())
            >>> print(t.shape)
            [165]

    """
    dtype = parameters[0].dtype
    origin_shapes = []
    for param in parameters:
        origin_shapes.append(param.shape)
        _inplace_reshape_dygraph(param, [-1])

    out = _create_tensor(dtype=dtype)
    if in_dygraph_mode():
        with paddle.base.dygraph.no_grad():
            tmp = _C_ops.concat(parameters, 0)
            tmp._share_underline_tensor_to(out)
    else:
        _dygraph_tracer().trace_op(
            type='concat',
            inputs={'X': parameters},
            outputs={'Out': [out]},
            attrs={'axis': 0},
            stop_gradient=True,
        )
    for i, param in enumerate(parameters):
        _inplace_reshape_dygraph(param, origin_shapes[i])
    out.stop_gradient = False
    return out


@dygraph_only
def vector_to_parameters(
    vec: Tensor, parameters: Iterable[Tensor], name: str | None = None
) -> None:
    """
    Transform a 1-D Tensor to the input ``parameters`` .

    Args:
        vec (Tensor): A 1-D Tensor, which will be sliced and copied to the input ``parameters`` .
        parameters (Iterable[Tensor]): Iterable Tensors that are trainable parameters of a Layer.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
       .. code-block:: python

            >>> import paddle
            >>> weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(3.))
            >>> linear1 = paddle.nn.Linear(10, 15, weight_attr)

            >>> vec = paddle.nn.utils.parameters_to_vector(linear1.parameters())

            >>> linear2 = paddle.nn.Linear(10, 15)
            >>> # copy weight of linear1 to linear2
            >>> paddle.nn.utils.vector_to_parameters(vec, linear2.parameters())
            >>> print((linear1.weight == linear2.weight).all())
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            True)
    """
    origin_shapes = []
    sections = []
    for param in parameters:
        shape = param.shape
        origin_shapes.append(shape)
        numel = reduce(lambda x, y: x * y, shape, 1)
        sections.append(numel)

    if len(sections) == 1:
        sections.append(0)

    if in_dygraph_mode():
        with paddle.base.dygraph.no_grad():
            res = _C_ops.split(vec, sections, 0)
            for i in range(0, len(parameters)):
                res[i]._share_underline_tensor_to(parameters[i])
    else:
        _dygraph_tracer().trace_op(
            type='split',
            inputs={'X': [vec]},
            outputs={'Out': parameters},
            attrs={'axis': 0, 'sections': sections},
            stop_gradient=True,
        )

    for i, param in enumerate(parameters):
        _inplace_reshape_dygraph(param, origin_shapes[i])
