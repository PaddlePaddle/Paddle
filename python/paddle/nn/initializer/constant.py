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

from paddle import _C_ops

from ...base import core, framework
from ...base.framework import (
    _current_expected_place,
    in_dygraph_mode,
    in_dynamic_or_pir_mode,
)

# TODO: define the initializers of Constant in neural network
from .initializer import Initializer

__all__ = []


class ConstantInitializer(Initializer):
    """Implements the constant initializer

    Args:
        value (float32, optional): constant value to initialize the variable. Default: 0.0.

    """

    def __init__(self, value=0.0, force_cpu=False):
        assert value is not None
        super().__init__()
        self._value = value
        self._force_cpu = force_cpu

    def forward(self, var, block=None):
        """Initialize the input tensor with constant.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        import paddle

        block = self._check_block(block)

        assert isinstance(
            var,
            (
                framework.Variable,
                framework.EagerParamBase,
                paddle.pir.Value,
                paddle.pir.core.ParameterMeta,
            ),
        )
        assert isinstance(block, (framework.Block, paddle.pir.Block))

        if in_dynamic_or_pir_mode():
            place = _current_expected_place()
            if self._force_cpu:
                place = core.CPUPlace()
            if in_dygraph_mode():
                if isinstance(var, framework.EagerParamBase) and var.is_dist():
                    out_var = _C_ops.full(
                        var._local_shape, float(self._value), var.dtype, place
                    )
                    out_var = (
                        paddle.distributed.auto_parallel.api.dtensor_from_local(
                            out_var, var.process_mesh, var.placements
                        )
                    )
                    out_var._share_underline_tensor_to(var)
                else:
                    _C_ops.full_(
                        var, var.shape, float(self._value), var.dtype, place
                    )
                return None
            else:
                return _C_ops.full(
                    var.shape, float(self._value), var.dtype, place
                )
        else:
            op = block.append_op(
                type="fill_constant",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "dtype": int(var.dtype),
                    "value": float(self._value),
                    'str_value': str(float(self._value)),
                    'force_cpu': self._force_cpu,
                },
                stop_gradient=True,
            )

            var.op = op
            return op


class Constant(ConstantInitializer):
    """Implement the constant initializer.

    Args:
        value (float32|float64, optional): constant value to initialize the parameter. Default: 0.0.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> paddle.seed(2023)
            >>> data = paddle.rand([30, 10, 2], dtype='float32')
            >>> linear = nn.Linear(2,
            ...                     4,
            ...                     weight_attr=nn.initializer.Constant(value=2.0))
            >>> res = linear(data)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[2., 2., 2., 2.],
             [2., 2., 2., 2.]])

    """

    def __init__(self, value=0.0):
        if value is None:
            raise ValueError("value must not be none.")
        super().__init__(value=value, force_cpu=False)
