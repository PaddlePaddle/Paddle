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

from ...fluid.initializer import XavierInitializer

__all__ = ['XavierNormal', 'XavierUniform']


class XavierNormal(XavierInitializer):
    """
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio, using a uniform distribution.

    The mean is 0 and the standard deviation is

    .. math::

        \sqrt{\\frac{2.0}{fan\_in + fan\_out}}


    Args:
        fan_in (float,default None): fan_in for Xavier initialization. If None, it is
                inferred from the variable.
        fan_out (float,default None): fan_out for Xavier initialization. If None, it is
                 inferred from the variable.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.XavierNormal())
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.XavierNormal())
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)
    """

    def __init__(self, fan_in=None, fan_out=None, name=None):
        super(XavierNormal, self).__init__()
        self._uniform = False
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = 0


class XavierUniform(XavierInitializer):
    """
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.

    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where

    .. math::

        x = \sqrt{\\frac{6.0}{fan\_in + fan\_out}}

    Args:
        fan_in (float,default None): fan_in for Xavier initialization. If None, it is
                inferred from the variable.
        fan_out (float,default None): fan_out for Xavier initialization. If None, it is
                 inferred from the variable.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.XavierUniform())
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.XavierUniform())
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)
    """

    def __init__(self, fan_in=None, fan_out=None, name=None):
        super(XavierUniform, self).__init__()
        self._uniform = True
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = 0
