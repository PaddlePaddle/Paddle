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

from ...fluid.initializer import UniformInitializer

__all__ = ['Uniform']


class Uniform(UniformInitializer):
    """Implements the random uniform distribution initializer

    Args:
        low (float): lower boundary of the uniform distribution
        high (float): upper boundary of the uniform distribution
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)
    """

    def __init__(self, low=-1.0, high=1.0, name=None):
        assert low is not None
        assert high is not None
        assert high >= low
        super(Uniform, self).__init__()
        self._low = low
        self._high = high
        self._seed = 0
        self._diag_num = 0
        self._diag_step = 0
        self._diag_val = 1.0
