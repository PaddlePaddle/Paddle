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

from ...fluid.initializer import NumpyArrayInitializer

__all__ = ['Assign']


class Assign(NumpyArrayInitializer):
    """Init an parameter with an numpy array
    This op initialize the variable by numpy array.

    Args:
        value (numpy): numpy array to initialize the variable
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor variable initialized by numpy.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            data = paddle.ones(shape=[1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(np.array([2,2])))
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(np.array([2])))
            linear = paddle.nn.Linear(2,2,weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)
    """

    pass
