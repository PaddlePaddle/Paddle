#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define the initializers of Kaiming functions in neural network
from ...fluid.initializer import MSRAInitializer

__all__ = []


class KaimingNormal(MSRAInitializer):
    r"""Implements the Kaiming Normal initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \frac{gain}{\sqrt{{fan\_in}}}

    Args:
        fan_in (float32|None, optional): fan_in (in_features) of trainable Tensor, If None, it will be infered automaticly. If you don't want to use in_features of the Tensor, you can set the value of 'fan_in' smartly by yourself. default is None.
        negative_slope (float, optional): negative_slope (only used with leaky_relu). default is 0.0.
        nonlinearity(str, optional): the non-linear function. default is relu.

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            linear = nn.Linear(2,
                               4,
                               weight_attr=nn.initializer.KaimingNormal())
            data = paddle.rand([30, 10, 2], dtype='float32')
            res = linear(data)

    """

    def __init__(self, fan_in=None, negative_slope=0.0, nonlinearity='relu'):
        super(KaimingNormal, self).__init__(uniform=False,
                                            fan_in=fan_in,
                                            seed=0,
                                            negative_slope=negative_slope,
                                            nonlinearity=nonlinearity)


class KaimingUniform(MSRAInitializer):
    r"""Implements the Kaiming Uniform initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.

    In case of Uniform distribution, the range is [-x, x], where

    .. math::

        x = gain \times \sqrt{\frac{3}{fan\_in}}

    Args:
        fan_in (float32|None, optional): fan_in (in_features) of trainable Tensor, If None, it will be infered automaticly. If you don't want to use in_features of the Tensor, you can set the value of 'fan_in' smartly by yourself. default is None.
        negative_slope (float, optional): negative_slope (only used with leaky_relu). default is 0.0.
        nonlinearity(str, optional): the non-linear function. default is relu.

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            linear = nn.Linear(2,
                               4,
                               weight_attr=nn.initializer.KaimingUniform())
            data = paddle.rand([30, 10, 2], dtype='float32')
            res = linear(data)

    """

    def __init__(self, fan_in=None, negative_slope=0.0, nonlinearity='relu'):
        super(KaimingUniform, self).__init__(uniform=True,
                                             fan_in=fan_in,
                                             seed=0,
                                             negative_slope=negative_slope,
                                             nonlinearity=nonlinearity)
