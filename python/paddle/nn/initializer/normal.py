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

from ...fluid.initializer import NormalInitializer
from ...fluid.initializer import TruncatedNormalInitializer

__all__ = ['Normal', 'TruncatedNormal']


class Normal(NormalInitializer):
    """Implements the Random Normal(Gaussian) distribution initializer

    Args:
        mean (float): mean of the normal distribution
        std (float): standard deviation of the normal distribution
        seed (int): random seed
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0))
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)
    """

    def __init__(self, mean=0.0, std=1.0, name=None):
        assert mean is not None
        assert std is not None
        super(Normal, self).__init__()
        self._mean = mean
        self._std_dev = std
        self._seed = 0


class TruncatedNormal(TruncatedNormalInitializer):
    """Implements the Random TruncatedNormal(Gaussian) distribution initializer

    Args:
        mean (float): mean of the normal distribution
        std (float): standard deviation of the normal distribution
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)
    """

    def __init__(self, mean=0.0, std=1.0, name=None):
        assert mean is not None
        assert std is not None
        super(TruncatedNormal, self).__init__()
        self._mean = mean
        self._std_dev = std
        self._seed = 0
