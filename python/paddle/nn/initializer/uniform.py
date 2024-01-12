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

from paddle import _C_ops, pir

from ...base import core, framework, unique_name
from ...base.data_feeder import check_variable_and_dtype
from ...base.framework import (
    _current_expected_place,
    in_dygraph_mode,
    in_pir_mode,
)
from .initializer import Initializer

__all__ = []


class UniformInitializer(Initializer):
    """Implements the random uniform distribution initializer

    Args:
        low (float, optional): Lower boundary of the uniform distribution. Default is :math:`-1.0`.
        high (float, optional): Upper boundary of the uniform distribution. Default is :math:`1.0`.
        seed (int, optional): Random seed. Default is 0.
        diag_num (int, optional): the number of diagonal elements to initialize.
            If set to 0, diagonal initialization will be not performed. Default is 0.
        diag_step (int, optional): Step size between two diagonal elements,
            which is generally the width of the square matrix. Default is 0.
        diag_val (float, optional): the value of the diagonal element to be initialized,
            default 1.0. It takes effect only if the diag_num is greater than 0. Default is :math:`1.0`.

    """

    def __init__(
        self, low=-1.0, high=1.0, seed=0, diag_num=0, diag_step=0, diag_val=1.0
    ):
        assert low is not None
        assert high is not None
        assert high >= low
        assert seed is not None
        assert diag_num is not None
        assert diag_step is not None
        assert diag_val is not None
        if diag_num > 0 or diag_step > 0:
            assert diag_num > 0 and diag_step > 0
        super().__init__()
        self._low = low
        self._high = high
        self._seed = seed
        self._diag_num = diag_num
        self._diag_step = diag_step
        self._diag_val = diag_val

    def forward(self, var, block=None):
        """Initialize the input tensor with Uniform distribution.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        assert not (
            isinstance(var, framework.EagerParamBase) and var.is_dist()
        ), "Currently, uniform initializer not support lazy init for dist param."
        block = self._check_block(block)

        assert isinstance(block, (framework.Block, pir.Block))
        if not in_dygraph_mode():
            check_variable_and_dtype(
                var,
                "Out",
                ["uint16", "float16", "float32", "float64"],
                "uniform_random",
            )

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initializers
        if var.dtype == core.VarDesc.VarType.FP16:
            out_dtype = core.VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['uniform_random', var.name, 'tmp'])
                ),
                shape=var.shape,
                dtype=out_dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
            )
        else:
            out_dtype = var.dtype
            out_var = var

        if in_dygraph_mode():
            out_var = _C_ops.uniform(
                var.shape,
                out_dtype,
                self._low,
                self._high,
                self._seed,
                _current_expected_place(),
            )
            if var.dtype == core.VarDesc.VarType.FP16:
                var_tmp = _C_ops.cast(out_var, var.dtype)
                var_tmp._share_underline_tensor_to(var)
            else:
                out_var._share_underline_tensor_to(var)
            return None
        elif in_pir_mode():
            if var.dtype == core.DataType.FLOAT16:
                out_dtype = core.DataType.FLOAT32
            else:
                out_dtype = var.dtype
            out_var = _C_ops.uniform(
                var.shape,
                out_dtype,
                self._low,
                self._high,
                self._seed,
                _current_expected_place(),
            )
            if (
                var.dtype == core.DataType.FLOAT16
                and out_var.dtype != core.DataType.FLOAT16
            ):
                return _C_ops.cast(out_var, var.dtype)
            return out_var
        else:
            op = block.append_op(
                type="uniform_random",
                inputs={},
                outputs={"Out": out_var},
                attrs={
                    "shape": var.shape,
                    "dtype": out_dtype,
                    "min": self._low,
                    "max": self._high,
                    "seed": self._seed,
                    "diag_num": self._diag_num,
                    "diag_step": self._diag_step,
                    "diag_val": self._diag_val,
                },
                stop_gradient=True,
            )

            if var.dtype == core.VarDesc.VarType.FP16:
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={"in_dtype": out_var.dtype, "out_dtype": var.dtype},
                )

            var.op = op
            return op


class Uniform(UniformInitializer):
    """The uniform distribution initializer.

    Args:
        low (float, optional): Lower boundary of the uniform distribution. Default is :math:`-1.0`.
        high (float, optional): Upper boundary of the uniform distribution. Default is :math:`1.0`.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A parameter initialized by uniform distribution.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(1)
            >>> data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            >>> weight_attr = paddle.framework.ParamAttr(
            ...     name="linear_weight",
            ...     initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
            >>> bias_attr = paddle.framework.ParamAttr(
            ...     name="linear_bias",
            ...     initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
            >>> linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[-0.48212373,  0.26492310],
             [ 0.17605734, -0.45379421]])

            >>> print(linear.bias)
            Parameter containing:
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [-0.11236754,  0.46462214])

            >>> res = linear(data)
            >>> print(res)
            Tensor(shape=[3, 1, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[[-0.41843393,  0.27575102]],
             [[-0.41843393,  0.27575102]],
             [[-0.41843393,  0.27575102]]])
    """

    def __init__(self, low=-1.0, high=1.0, name=None):
        assert low is not None, 'low should not be None'
        assert high is not None, 'high should not be None'
        assert high >= low, 'high should greater or equal than low'
        super().__init__(
            low=low, high=high, seed=0, diag_num=0, diag_step=0, diag_val=1.0
        )
