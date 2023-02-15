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

import math

from paddle import _C_ops

from ...fluid import core, framework, unique_name
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.framework import _current_expected_place, in_dygraph_mode
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

        x = \sqrt{\\frac{6.0}{fan\_in + fan\_out}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \sqrt{\\frac{2.0}{fan\_in + fan\_out}}


    Args:
        uniform (bool, optional): whether to use uniform ,if False use normal distribution. Default is True.
        fan_in (float, optional): fan_in for Xavier initialization. If None, it is
                inferred from the variable. Default is None.
        fan_out (float, optional): fan_out for Xavier initialization. If None, it is
                 inferred from the variable. Default is None.
        seed (int, optional): Random seed. Default is 0.

    Note:
        It is recommended to set fan_in and fan_out to None for most cases.

    """

    def __init__(self, uniform=True, fan_in=None, fan_out=None, seed=0):
        assert uniform is not None
        assert seed is not None
        super().__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = seed

    def forward(self, var, block=None):
        """Initialize the input tensor with Xavier initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(block, framework.Block)
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

        # to be compatible of fp16 initalizers
        if var.dtype == core.VarDesc.VarType.FP16 or (
            var.dtype == core.VarDesc.VarType.BF16 and not self._uniform
        ):
            out_dtype = core.VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['xavier_init', var.name, 'tmp'])
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
            if self._uniform:
                limit = math.sqrt(6.0 / float(fan_in + fan_out))
                out_var = _C_ops.uniform(
                    out_var.shape,
                    out_dtype,
                    -limit,
                    limit,
                    self._seed,
                    _current_expected_place(),
                )
            else:
                std = math.sqrt(2.0 / float(fan_in + fan_out))

                place = _current_expected_place()
                out_var = _C_ops.gaussian(
                    out_var.shape, 0.0, std, self._seed, out_dtype, place
                )

            if var.dtype == core.VarDesc.VarType.FP16 or (
                var.dtype == core.VarDesc.VarType.BF16 and not self._uniform
            ):
                var_tmp = _C_ops.cast(out_var, var.dtype)
                var_tmp._share_underline_tensor_to(var)
            else:
                out_var._share_underline_tensor_to(var)
            return None
        else:
            if self._uniform:
                limit = math.sqrt(6.0 / float(fan_in + fan_out))
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
                std = math.sqrt(2.0 / float(fan_in + fan_out))
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

            if var.dtype == core.VarDesc.VarType.FP16 or (
                var.dtype == core.VarDesc.VarType.BF16 and not self._uniform
            ):
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={"in_dtype": out_var.dtype, "out_dtype": var.dtype},
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

        \sqrt{\frac{2.0}{fan\_in + fan\_out}}.


    Args:
        fan_in (float, optional): fan_in for Xavier initialization, which is
                inferred from the Tensor. Default is None.
        fan_out (float, optional): fan_out for Xavier initialization, which is
                 inferred from the Tensor. Default is None.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A parameter initialized by Xavier weight, using a normal distribution.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.XavierNormal())
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.XavierNormal())
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # inear.weight:  [[ 0.06910077 -0.18103665]
            #                 [-0.02546741 -1.0402188 ]]
            # linear.bias:  [-0.5012929   0.12418364]

            res = linear(data)
            # res:  [[[-0.4576595 -1.0970719]]
            #        [[-0.4576595 -1.0970719]]
            #        [[-0.4576595 -1.0970719]]]
    """

    def __init__(self, fan_in=None, fan_out=None, name=None):
        super().__init__(uniform=False, fan_in=fan_in, fan_out=fan_out, seed=0)


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

        x = \sqrt{\frac{6.0}{fan\_in + fan\_out}}.

    Args:
        fan_in (float, optional): fan_in for Xavier initialization, which is
                inferred from the Tensor. Default is None.
        fan_out (float, optional): fan_out for Xavier initialization, which is
                 inferred from the Tensor. Default is None.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A parameter initialized by Xavier weight, using a uniform distribution.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.XavierUniform())
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.XavierUniform())
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # linear.weight:  [[-0.04229349 -1.1248565 ]
            #                  [-0.10789523 -0.5938053 ]]
            # linear.bias:  [ 1.1983747  -0.40201235]

            res = linear(data)
            # res:  [[[ 1.0481861 -2.1206741]]
            #        [[ 1.0481861 -2.1206741]]
            #        [[ 1.0481861 -2.1206741]]]
    """

    def __init__(self, fan_in=None, fan_out=None, name=None):
        super().__init__(uniform=True, fan_in=fan_in, fan_out=fan_out, seed=0)
