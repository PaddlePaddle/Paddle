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

from typing import TYPE_CHECKING, Any

import paddle
from paddle import _C_ops

from ...base import core, framework, unique_name
from ...base.data_feeder import check_type
from ...base.framework import (
    _current_expected_place,
    in_dygraph_mode,
    in_pir_mode,
)
from .initializer import Initializer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = []


class NumpyArrayInitializer(Initializer):
    """Init an parameter with an numpy array
    This api initialize the tensor by numpy array.

    Args:
        value (numpy): numpy array to initialize the tensor

    Returns:
        A Tensor initialized by numpy.

    """

    def __init__(self, value: npt.NDArray[Any]) -> None:
        import numpy

        assert isinstance(value, numpy.ndarray)
        super().__init__()
        self._value = value

    def forward(
        self, var: paddle.Tensor, block: paddle.pir.Block | None = None
    ) -> paddle.Tensor | None:
        """Initialize the input tensor with Numpy array.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block|None, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        assert not (
            isinstance(var, framework.EagerParamBase) and var.is_dist()
        ), "Currently, assign initializer not support lazy init for dist param."
        block = self._check_block(block)

        assert isinstance(
            var, (framework.Variable, paddle.pir.core.ParameterMeta)
        )
        assert isinstance(block, (framework.Block, paddle.pir.Block))

        # to be compatible of fp16 initializers
        origin_dtype = var.dtype
        if origin_dtype in [
            core.VarDesc.VarType.FP16,
            core.VarDesc.VarType.BF16,
        ]:
            out_dtype = core.VarDesc.VarType.FP32
            np_value = self._value.astype("float32")
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['numpy_array_init', var.name, 'tmp'])
                ),
                shape=var.shape,
                dtype=out_dtype,
                type=core.VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
            )
        elif origin_dtype in [core.DataType.FLOAT16, core.DataType.BFLOAT16]:
            out_var = var
            out_dtype = core.DataType.FLOAT32
            np_value = self._value.astype("float32")
        else:
            out_var = var
            out_dtype = origin_dtype
            np_value = self._value

        if out_dtype in (core.VarDesc.VarType.FP32, core.DataType.FLOAT32):
            value_name = "values"
            values = [float(v) for v in np_value.flat]
        elif out_dtype in (core.VarDesc.VarType.FP64, core.DataType.FLOAT64):
            value_name = "values"
            values = [float(v) for v in np_value.flat]
        elif out_dtype in (core.VarDesc.VarType.INT32, core.DataType.INT32):
            value_name = "values"
            values = [int(v) for v in np_value.flat]
        elif out_dtype in (
            core.VarDesc.VarType.INT8,
            core.VarDesc.VarType.UINT8,
            core.DataType.INT8,
            core.DataType.UINT8,
        ):
            value_name = "int8_values"
            values = [int(v) for v in np_value.flat]
        else:
            raise ValueError(f"Unsupported dtype {self._value.dtype}")
        if self._value.size > 1024 * 1024 * 1024:
            raise ValueError(
                "The size of input is too big. Please consider "
                "saving it to file and 'load_op' to load it"
            )

        if in_dygraph_mode():
            _C_ops.assign_value_(
                out_var,
                list(self._value.shape),
                out_dtype,
                values,
                _current_expected_place(),
            )
            if origin_dtype in [
                core.VarDesc.VarType.FP16,
                core.VarDesc.VarType.BF16,
                core.DataType.FLOAT16,
                core.DataType.BFLOAT16,
            ]:
                var_tmp = _C_ops.cast(out_var, origin_dtype)
                var_tmp._share_underline_tensor_to(var)
            else:
                out_var._share_underline_tensor_to(var)
            return None
        elif in_pir_mode():
            out_var = _C_ops.assign_value(
                list(self._value.shape),
                out_dtype,
                values,
                _current_expected_place(),
            )
            if origin_dtype in [core.DataType.FLOAT16, core.DataType.BFLOAT16]:
                out_var = _C_ops.cast(out_var, origin_dtype)
            return out_var
        else:
            op = block.append_op(
                type='assign_value',
                outputs={'Out': out_var},
                attrs={
                    'dtype': out_dtype,
                    'shape': list(self._value.shape),
                    value_name: values,
                },
                stop_gradient=True,
            )

            if origin_dtype in [
                core.VarDesc.VarType.FP16,
                core.VarDesc.VarType.BF16,
            ]:
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={
                        "in_dtype": out_var.dtype,
                        "out_dtype": origin_dtype,
                    },
                )

            var.op = op
            return op


class Assign(NumpyArrayInitializer):
    """Init an parameter with a numpy array, list, or tensor.

    Args:
        value (Tensor|numpy.ndarray|list|tuple): numpy array, list, tuple, or tensor to initialize the parameter.
        name(str|None, optional): Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        A parameter initialized by the input numpy array, list, or tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> # numpy array
            >>> data_1 = paddle.ones(shape=[1, 2], dtype='float32')
            >>> weight_attr_1 = paddle.ParamAttr(
            ...     name="linear_weight_1",
            ...     initializer=paddle.nn.initializer.Assign(np.array([2, 2])))
            >>> bias_attr_1 = paddle.ParamAttr(
            ...     name="linear_bias_1",
            ...     initializer=paddle.nn.initializer.Assign(np.array([2])))
            >>> linear_1 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
            >>> print(linear_1.weight.numpy())
            [2. 2.]
            >>> print(linear_1.bias.numpy())
            [2.]

            >>> res_1 = linear_1(data_1)
            >>> print(res_1.numpy())
            [6.]

            >>> # python list
            >>> data_2 = paddle.ones(shape=[1, 2], dtype='float32')
            >>> weight_attr_2 = paddle.ParamAttr(
            ...     name="linear_weight_2",
            ...     initializer=paddle.nn.initializer.Assign([2, 2]))
            >>> bias_attr_2 = paddle.ParamAttr(
            ...     name="linear_bias_2",
            ...     initializer=paddle.nn.initializer.Assign([2]))
            >>> linear_2 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_2, bias_attr=bias_attr_2)
            >>> print(linear_2.weight.numpy())
            [2. 2.]
            >>> print(linear_2.bias.numpy())
            [2.]

            >>> res_2 = linear_2(data_2)
            >>> print(res_2.numpy())
            [6.]

            >>> # tensor
            >>> data_3 = paddle.ones(shape=[1, 2], dtype='float32')
            >>> weight_attr_3 = paddle.ParamAttr(
            ...     name="linear_weight_3",
            ...     initializer=paddle.nn.initializer.Assign(paddle.full([2], 2)))
            >>> bias_attr_3 = paddle.ParamAttr(
            ...     name="linear_bias_3",
            ...     initializer=paddle.nn.initializer.Assign(paddle.full([1], 2)))
            >>> linear_3 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_3, bias_attr=bias_attr_3)
            >>> print(linear_3.weight.numpy())
            [2. 2.]
            >>> print(linear_3.bias.numpy())
            [2.]

            >>> res_3 = linear_3(data_3)
            >>> print(res_3.numpy())
            [6.]
    """

    def __init__(
        self,
        value: npt.NDArray[Any] | Sequence[int] | paddle.Tensor,
        name: str | None = None,
    ) -> None:
        import numpy

        check_type(
            value,
            'value',
            (numpy.ndarray, list, tuple, paddle.static.Variable),
            'Assign',
        )

        if isinstance(value, (list, tuple)):
            value = numpy.array(value)

        # TODO: value is already is a tensor, accounting efficiency maybe it does not need to convert tensor to numpy data and then initialized.
        if isinstance(value, paddle.static.Variable):
            value = value.numpy(False)

        super().__init__(value)
