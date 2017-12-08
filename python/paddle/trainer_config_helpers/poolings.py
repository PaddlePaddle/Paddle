# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
"""

__all__ = [
    "BasePoolingType", "MaxPooling", "AvgPooling", "MaxWithMaskPooling",
    "CudnnMaxPooling", "CudnnAvgPooling", "CudnnAvgInclPadPooling",
    "SumPooling", "SquareRootNPooling"
]


class BasePoolingType(object):
    """
    Base Pooling Type.
    Note these pooling types are used for sequence input, not for images.
    Each PoolingType contains one parameter:

    :param name: pooling layer type name used by paddle.
    :type name: basestring

    """

    def __init__(self, name):
        self.name = name


class MaxPooling(BasePoolingType):
    """
    Max pooling.

    Return the very large values for each dimension in sequence or time steps.

    ..  math::

        max(samples\\_of\\_a\\_sequence)

    :param output_max_index: True if output sequence max index instead of max
                             value. None means use default value in proto.
    :type output_max_index: bool|None
    """

    def __init__(self, output_max_index=None):
        BasePoolingType.__init__(self, "max")
        self.output_max_index = output_max_index


class MaxWithMaskPooling(BasePoolingType):
    """
    MaxWithMask pooling.

    Not only return the very large values for each dimension in sequence or time steps,
    but also the location indices of found maxinum values.

    """

    def __init__(self):
        BasePoolingType.__init__(self, "max-pool-with-mask")


class CudnnMaxPooling(BasePoolingType):
    """
    Cudnn max pooling only support GPU. Return the maxinum value in the
    pooling window.
    """

    def __init__(self):
        BasePoolingType.__init__(self, "cudnn-max-pool")


class CudnnAvgPooling(BasePoolingType):
    """
    Cudnn average pooling only support GPU. Return the average value in the
    pooling window.
    """

    def __init__(self):
        BasePoolingType.__init__(self, "cudnn-avg-pool")


class CudnnAvgInclPadPooling(BasePoolingType):
    """
    Cudnn average pooling only support GPU. Return the average value in the
    pooling window taking into account the padding cells.
    """

    def __init__(self):
        BasePoolingType.__init__(self, "cudnn-avg-incl-pad-pool")


class AvgPooling(BasePoolingType):
    """
    Average pooling.

    Return the average values for each dimension in sequence or time steps.

    ..  math::

        sum(samples\\_of\\_a\\_sequence)/sample\\_num
    """
    STRATEGY_AVG = "average"
    STRATEGY_SUM = "sum"
    STRATEGY_SQROOTN = "squarerootn"

    def __init__(self, strategy=STRATEGY_AVG):
        BasePoolingType.__init__(self, "average")
        self.strategy = strategy


class SumPooling(AvgPooling):
    """
    Sum pooling.

    Return the sum values of each dimension in sequence or time steps.

    ..  math::

        sum(samples\\_of\\_a\\_sequence)
    """

    def __init__(self):
        AvgPooling.__init__(self, AvgPooling.STRATEGY_SUM)


class SquareRootNPooling(AvgPooling):
    """
    Square Root Pooling.

    Return the square root values of each dimension in sequence or time steps.

    ..  math::

        sum(samples\\_of\\_a\\_sequence)/sqrt(sample\\_num)
    """

    def __init__(self):
        AvgPooling.__init__(self, AvgPooling.STRATEGY_SQROOTN)
