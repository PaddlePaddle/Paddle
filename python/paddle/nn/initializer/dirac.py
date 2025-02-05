#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _C_ops, in_dynamic_mode, pir
from paddle.utils import unique_name

from ... import base
from ...base import core, framework
from ...base.core import VarDesc
from ...base.data_feeder import check_variable_and_dtype
from ...base.framework import _current_expected_place
from .initializer import Initializer

__all__ = []


class Dirac(Initializer):
    r"""Initialize the 3D/4D/5D Tensor with Dirac delta function.

    It can reserve the feature of convolution layer input, which means that
    as many channels are reserved as possible.

    In this initialize method, elements in the middle of convolution kernels will
    be set to 1 . The formula can be described as follow.

    .. math::

        X[d, d, shape[2]//2, shape[3]//2, ...]=1,  \   d=0,1...N

    where, ``N`` is the minimum value of ``in_channels`` and ``out_channels``

    Args:
        groups(int|None, optional): 0-dimension of the Tensor will be divided by groups,
            each group has the same value. Default: 1.
        name(str|None, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Dirac initializer instance objects.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # 1. For kernel_size is uneven number:
            >>> attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
            >>> conv = paddle.nn.Conv1D(3, 2, 3, weight_attr=attr)
            >>> print(conv.weight)
            Parameter containing:
            Tensor(shape=[2, 3, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
            [[[0., 1., 0.],
              [0., 0., 0.],
              [0., 0., 0.]],
             [[0., 0., 0.],
              [0., 1., 0.],
              [0., 0., 0.]]])
            >>> input = paddle.rand([8, 3, 10])
            >>> output = conv(input)
            >>> output == input[:, 0:2, 1:9]
            >>> print(output.shape)
            [8, 2, 8]
            >>> # It means output is almost the same with input, 2 channels are reserved

            >>> # 2. For kernel_size is even number:
            >>> attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
            >>> conv = paddle.nn.Conv1D(3, 2, 4, weight_attr=attr)
            >>> print(conv.weight)
            Parameter containing:
            Tensor(shape=[2, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=False,
            [[[0., 0., 1., 0.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]],
             [[0., 0., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 0.]]])
    """

    def __init__(self, groups: int = 1, name: str | None = None) -> None:
        assert groups > 0 and isinstance(
            groups, int
        ), " 'groups' must be a positive integer. "
        super().__init__()
        self._groups = groups

    def __call__(
        self, var: paddle.Tensor, block: pir.Block | None = None
    ) -> paddle.Tensor:
        """Initialize the input tensor with dirac initializer.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block|None, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The most critical OP(scatter) in this initializer, which contains 7~8 ops in total.
        """
        assert not (
            isinstance(var, framework.EagerParamBase) and var.is_dist()
        ), "Currently, dirac initializer not support lazy init for dist param."
        block = self._check_block(block)
        assert isinstance(var, (framework.Variable, pir.core.ParameterMeta))
        assert isinstance(block, (framework.Block, pir.Block))
        check_variable_and_dtype(
            var, "Out", ['float16', 'bfloat16', 'float32', 'float64'], 'Dirac'
        )

        assert len(var.shape) in [
            3,
            4,
            5,
        ], "Only Tensor with 3/4/5 dimensions can be initialized by Dirac"
        assert (
            var.shape[0] % self._groups
        ) == 0, "Tensor 0-dimension must be divisible by groups"

        if framework.in_pir_mode():
            if var.dtype != core.DataType.FLOAT32:
                out_dtype = core.DataType.FLOAT32
                out_var = var
            else:
                out_dtype = var.dtype
                out_var = var
        else:
            if var.dtype != VarDesc.VarType.FP32:
                out_dtype = VarDesc.VarType.FP32
                out_var = block.create_var(
                    name=unique_name.generate(
                        ".".join(['dirac', var.name, 'tmp'])
                    ),
                    shape=var.shape,
                    dtype=out_dtype,
                    type=VarDesc.VarType.DENSE_TENSOR,
                    persistable=False,
                )
            else:
                out_dtype = var.dtype
                out_var = var

        op = None
        if framework.in_dygraph_mode():
            with base.dygraph.no_grad():
                place = _current_expected_place()
                _C_ops.full_(
                    out_var, out_var.shape, str(float(0)), out_dtype, place
                )
        elif framework.in_pir_mode():
            place = _current_expected_place()
            out_var = _C_ops.full(out_var.shape, float(0), out_dtype, place)
        else:
            block.append_op(
                type='fill_constant',
                inputs={},
                outputs={'Out': out_var},
                attrs={
                    'value': float(0),
                    'dtype': out_var.dtype,
                    'shape': out_var.shape,
                },
                stop_gradient=True,
            )

        origin_shape = var.shape
        num_per_group = origin_shape[0] // self._groups
        min_shape = min(num_per_group, origin_shape[1])

        idx_list = []
        value_list = []
        strides = []
        prod = 1
        for dim in reversed(origin_shape):
            strides.insert(0, prod)
            prod *= dim
        for i in range(self._groups):
            for j in range(min_shape):
                value_list.append(1.0)
                offset = 0
                for k, stride in enumerate(strides):
                    if k == 0:
                        offset += (j + i * num_per_group) * stride
                    elif k == 1:
                        offset += j * stride
                    else:
                        offset += origin_shape[k] // 2 * stride
                idx_list.append(offset)
        if framework.in_dygraph_mode():
            with base.dygraph.no_grad():
                tmp_out = _C_ops.reshape(out_var, [-1])
                tmp_out._share_underline_tensor_to(out_var)
        elif framework.in_pir_mode():
            out_var = _C_ops.reshape(out_var, [-1])
        else:
            x_shape = block.create_var(
                name=unique_name.generate(".".join([out_var.name, "XShape"])),
                dtype=out_dtype,
                shape=out_var.shape,
                type=VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
                stop_gradient=True,
            )
            block.append_op(
                type="reshape2",
                inputs={"X": out_var},
                attrs={'shape': [-1]},
                outputs={"Out": out_var, "XShape": x_shape},
                stop_gradient=True,
            )

        if framework.in_pir_mode():
            index_tensor = paddle.zeros(
                [len(idx_list)], dtype=core.DataType.INT64
            )
            index_tensor.stop_gradient = True
        else:
            index_tensor = block.create_var(
                name=unique_name.generate('scatter_index'),
                persistable=False,
                stop_gradient=True,
            )

        if framework.in_dygraph_mode():
            with base.dygraph.no_grad():
                tmp_tensor = framework._create_tensor()
                _C_ops.assign_value_(
                    tmp_tensor,
                    [len(idx_list)],
                    VarDesc.VarType.INT64,
                    idx_list,
                    _current_expected_place(),
                )
                tmp_tensor._share_underline_tensor_to(index_tensor)
        elif framework.in_pir_mode():
            _C_ops.assign_value_(
                index_tensor,
                [len(idx_list)],
                core.DataType.INT64,
                idx_list,
                _current_expected_place(),
            )
        else:
            block.append_op(
                type='assign_value',
                outputs={'Out': index_tensor},
                attrs={
                    'dtype': VarDesc.VarType.INT64,
                    'shape': [len(idx_list)],
                    'values': idx_list,
                },
                stop_gradient=True,
            )
        if framework.in_pir_mode():
            value_tensor = paddle.zeros(
                [len(value_list)], dtype=core.DataType.FLOAT32
            )
            value_tensor.stop_gradient = True
        else:
            value_tensor = block.create_var(
                name=unique_name.generate('scatter_value'),
                persistable=False,
                stop_gradient=True,
            )

        if framework.in_dygraph_mode():
            with base.dygraph.no_grad():
                tmp_tensor = framework._create_tensor()
                _C_ops.assign_value_(
                    tmp_tensor,
                    [len(value_list)],
                    VarDesc.VarType.FP32,
                    value_list,
                    _current_expected_place(),
                )

                tmp_tensor._share_underline_tensor_to(value_tensor)
        elif framework.in_pir_mode():
            _C_ops.assign_value_(
                value_tensor,
                [len(value_list)],
                core.DataType.FLOAT32,
                value_list,
                _current_expected_place(),
            )
        else:
            block.append_op(
                type='assign_value',
                outputs={'Out': value_tensor},
                attrs={
                    'dtype': VarDesc.VarType.FP32,
                    'shape': [len(value_list)],
                    'values': value_list,
                },
                stop_gradient=True,
            )

        if framework.in_dygraph_mode():
            with base.dygraph.no_grad():
                tmp_out = _C_ops.scatter(
                    out_var, index_tensor, value_tensor, True
                )
                tmp_out._share_underline_tensor_to(out_var)
                tmp_reshape_out = _C_ops.reshape(out_var, origin_shape)
                tmp_reshape_out._share_underline_tensor_to(out_var)
                if var.dtype != VarDesc.VarType.FP32:
                    tmp_cast_out = _C_ops.cast(out_var, var.dtype)
                    tmp_cast_out._share_underline_tensor_to(var)
        elif framework.in_pir_mode():
            out_var = _C_ops.scatter(out_var, index_tensor, value_tensor, True)
            out_var = _C_ops.reshape(out_var, origin_shape)
            if var.dtype != core.DataType.FLOAT32:
                return _C_ops.cast(out_var, var.dtype)
            return out_var
        else:
            op = block.append_op(
                type="scatter",
                inputs={
                    "X": out_var,
                    "Ids": index_tensor,
                    "Updates": value_tensor,
                },
                attrs={'overwrite': True},
                outputs={"Out": out_var},
                stop_gradient=True,
            )
            x_shape = block.create_var(
                name=unique_name.generate(".".join([out_var.name, "XShape"])),
                dtype=out_dtype,
                shape=out_var.shape,
                type=VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
                stop_gradient=True,
            )
            block.append_op(
                type="reshape2",
                inputs={"X": out_var},
                attrs={'shape': origin_shape},
                outputs={"Out": out_var, "XShape": x_shape},
                stop_gradient=True,
            )
            if var.dtype != VarDesc.VarType.FP32:
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={"in_dtype": out_var.dtype, "out_dtype": var.dtype},
                    stop_gradient=True,
                )
        if not in_dynamic_mode():
            var.op = op
        return op
