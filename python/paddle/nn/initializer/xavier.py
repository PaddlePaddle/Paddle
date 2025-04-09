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

import math

import paddle
from paddle import _C_ops

from ...base import core, framework, unique_name
from ...base.data_feeder import check_variable_and_dtype
from ...base.framework import (
    _current_expected_place,
    in_dygraph_mode,
    in_pir_mode,
)
from .initializer import Initializer

__all__ = []


class XavierInitializer(Initializer):
    r"""
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.

    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where

    .. math::

        x = gain \times \sqrt{\\frac{6.0}{fan\_in + fan\_out}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

       gain \times \sqrt{\\frac{2.0}{fan\_in + fan\_out}}


    Args:
        uniform (bool, optional): whether to use uniform ,if False use normal distribution. Default is True.
        fan_in (float|None, optional): fan_in for Xavier initialization. If None, it is
                inferred from the variable. Default is None.
        fan_out (float|None, optional): fan_out for Xavier initialization. If None, it is
                 inferred from the variable. Default is None.
        seed (int, optional): Random seed. Default is 0.
        gain (float, optional): Scaling Tensor. Default is 1.0.

    Note:
        It is recommended to set fan_in and fan_out to None for most cases.

    """

    def __init__(
        self,
        uniform: bool = True,
        fan_in: float | None = None,
        fan_out: float | None = None,
        seed: int = 0,
        gain: float = 1.0,
    ) -> None:
        assert uniform is not None
        assert seed is not None
        super().__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = seed
        self._gain = gain

    def forward(
        self, var: paddle.Tensor, block: paddle.pir.Block | None = None
    ) -> paddle.Tensor | None:
        """Initialize the input tensor with Xavier initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block|None, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """

        block = self._check_block(block)
        assert isinstance(block, (framework.Block, paddle.pir.Block))
        if not isinstance(var, paddle.pir.core.ParameterMeta):
            check_variable_and_dtype(
                var,
                "Out",
                ["uint16", "float16", "float32", "float64"],
                "xavier_init",
            )

        f_in, f_out = self._compute_fans(var)

        # If fan_in and fan_out are passed, use them
        fan_in = f_in if self._fan_in is None else self._fan_in
        fan_out = f_out if self._fan_out is None else self._fan_out

        if self._seed == 0:
            self._seed = block.program.random_seed

        out_var_shape = (
            var._local_shape
            if (isinstance(var, framework.EagerParamBase) and var.is_dist())
            else var.shape
        )
        # to be compatible of fp16 initializers
        origin_dtype = var.dtype
        if origin_dtype == core.VarDesc.VarType.FP16 or (
            origin_dtype == core.VarDesc.VarType.BF16 and not self._uniform
        ):
            out_dtype = core.VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['xavier_init', var.name, 'tmp'])
                ),
                shape=out_var_shape,
                dtype=out_dtype,
                type=core.VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
            )
        elif (
            origin_dtype in (core.DataType.FLOAT16, core.DataType.BFLOAT16)
            and not self._uniform
        ):
            out_dtype = core.DataType.FLOAT32
            out_var = var
        else:
            out_dtype = origin_dtype
            out_var = var

        if in_dygraph_mode():
            if self._uniform:
                if 0 in [fan_in, fan_out]:
                    limit = 0.0
                else:
                    limit = self._gain * math.sqrt(
                        6.0 / float(fan_in + fan_out)
                    )
                out_var = _C_ops.uniform(
                    out_var_shape,
                    out_dtype,
                    -limit,
                    limit,
                    self._seed,
                    _current_expected_place(),
                )
            else:
                if 0 in [fan_in, fan_out]:
                    std = 0.0
                else:
                    std = self._gain * math.sqrt(2.0 / float(fan_in + fan_out))

                place = _current_expected_place()
                out_var = _C_ops.gaussian(
                    out_var_shape,
                    0.0,
                    std,
                    self._seed,
                    out_dtype,
                    place,
                )

            if origin_dtype == core.VarDesc.VarType.FP16 or (
                origin_dtype
                in [
                    core.VarDesc.VarType.BF16,
                    core.DataType.FLOAT16,
                    core.DataType.BFLOAT16,
                ]
                and not self._uniform
            ):
                out_var = _C_ops.cast(out_var, origin_dtype)
            if isinstance(var, framework.EagerParamBase) and var.is_dist():
                # lazy init for dist tensor
                out_var = (
                    paddle.distributed.auto_parallel.api.dtensor_from_local(
                        out_var, var.process_mesh, var.placements
                    )
                )
            out_var._share_underline_tensor_to(var)
            return None
        elif in_pir_mode():
            if self._uniform:
                if 0 in [fan_in, fan_out]:
                    limit = 0.0
                else:
                    limit = self._gain * math.sqrt(
                        6.0 / float(fan_in + fan_out)
                    )
                out_var = paddle._pir_ops.uniform(
                    out_var.shape,
                    out_dtype,
                    -limit,
                    limit,
                    self._seed,
                    _current_expected_place(),
                )
            else:
                if 0 in [fan_in, fan_out]:
                    std = 0.0
                else:
                    std = self._gain * math.sqrt(2.0 / float(fan_in + fan_out))
                out_var = _C_ops.gaussian(
                    out_var.shape,
                    0.0,
                    std,
                    self._seed,
                    out_dtype,
                    _current_expected_place(),
                )

            if (
                origin_dtype in (core.DataType.FLOAT16, core.DataType.BFLOAT16)
                and not self._uniform
            ):
                return _C_ops.cast(out_var, origin_dtype)

            return out_var
        else:
            if self._uniform:
                if 0 in [fan_in, fan_out]:
                    limit = 0.0
                else:
                    limit = self._gain * math.sqrt(
                        6.0 / float(fan_in + fan_out)
                    )
                op = block.append_op(
                    type="uniform_random",
                    inputs={},
                    outputs={"Out": out_var},
                    attrs={
                        "shape": out_var.shape,
                        "dtype": out_dtype,
                        "min": -limit,
                        "max": limit,
                        "seed": self._seed,
                    },
                    stop_gradient=True,
                )
            else:
                if 0 in [fan_in, fan_out]:
                    std = 0.0
                else:
                    std = self._gain * math.sqrt(2.0 / float(fan_in + fan_out))
                op = block.append_op(
                    type="gaussian_random",
                    outputs={"Out": out_var},
                    attrs={
                        "shape": out_var.shape,
                        "dtype": out_var.dtype,
                        "mean": 0.0,
                        "std": std,
                        "seed": self._seed,
                    },
                    stop_gradient=True,
                )

            if origin_dtype == core.VarDesc.VarType.FP16 or (
                origin_dtype == core.VarDesc.VarType.BF16 and not self._uniform
            ):
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


class XavierNormal(XavierInitializer):
    r"""
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio, using a normal distribution whose mean is :math:`0` and standard deviation is

    .. math::

        gain \times \sqrt{\frac{2.0}{fan\_in + fan\_out}}.


    Args:
        fan_in (float|None, optional): fan_in for Xavier initialization, which is
                inferred from the Tensor. Default is None.
        fan_out (float|None, optional): fan_out for Xavier initialization, which is
                 inferred from the Tensor. Default is None.
        gain (float, optional): Scaling Tensor. Default is 1.0.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A parameter initialized by Xavier weight, using a normal distribution.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(1)
            >>> data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            >>> weight_attr = paddle.framework.ParamAttr(
            ...     name="linear_weight",
            ...     initializer=paddle.nn.initializer.XavierNormal())
            >>> bias_attr = paddle.framework.ParamAttr(
            ...     name="linear_bias",
            ...     initializer=paddle.nn.initializer.XavierNormal())
            >>> linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[-0.21607460,  0.08382989],
             [ 0.29147008, -0.07049121]])

            >>> print(linear.bias)
            Parameter containing:
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [1.06076419, 0.87684733])

            >>> res = linear(data)
            >>> print(res)
            Tensor(shape=[3, 1, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[[1.13615966, 0.89018601]],
             [[1.13615966, 0.89018601]],
             [[1.13615966, 0.89018601]]])
    """

    def __init__(
        self,
        fan_in: float | None = None,
        fan_out: float | None = None,
        gain: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            uniform=False, fan_in=fan_in, fan_out=fan_out, seed=0, gain=gain
        )


class XavierUniform(XavierInitializer):
    r"""
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.

    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is :math:`[-x,x]`, where

    .. math::

        x = gain \times \sqrt{\frac{6.0}{fan\_in + fan\_out}}.

    Args:
        fan_in (float|None, optional): fan_in for Xavier initialization, which is
                inferred from the Tensor. Default is None.
        fan_out (float|None, optional): fan_out for Xavier initialization, which is
                 inferred from the Tensor. Default is None.
        gain (float, optional): Scaling Tensor. Default is 1.0.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A parameter initialized by Xavier weight, using a uniform distribution.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(1)
            >>> data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            >>> weight_attr = paddle.framework.ParamAttr(
            ...     name="linear_weight",
            ...     initializer=paddle.nn.initializer.XavierUniform())
            >>> bias_attr = paddle.framework.ParamAttr(
            ...     name="linear_bias",
            ...     initializer=paddle.nn.initializer.XavierUniform())
            >>> linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[-1.18095720,  0.64892638],
             [ 0.43125069, -1.11156428]])
            >>> print(linear.bias)
            Parameter containing:
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [-0.27524316,  1.13808715])

            >>> res = linear(data)
            >>> print(res)
            Tensor(shape=[3, 1, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[[-1.02494967,  0.67544925]],
             [[-1.02494967,  0.67544925]],
             [[-1.02494967,  0.67544925]]])
    """

    def __init__(
        self,
        fan_in: float | None = None,
        fan_out: float | None = None,
        gain: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            uniform=True, fan_in=fan_in, fan_out=fan_out, seed=0, gain=gain
        )
