#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ...fluid.dygraph import layers
from ...tensor import math, manipulation

__all__ = []


class FloatFunctionalLayer(layers.Layer):
    def __init__(self):
        super(FloatFunctionalLayer, self).__init__()

    def add(x, y, name=None):
        """
        Wrap paddle.add
        """
        return math.add(x, y, name)

    def subtract(x, y, name=None):
        """
        Wrap paddle.subtract
        """
        return math.subtract(x, y, name)

    def multiply(x, y, name=None):
        """
        Wrap paddle.multiply
        """
        return math.multiply(x, y, name)

    def divide(x, y, name=None):
        """
        Wrap paddle.divide
        """
        return math.divide(x, y, name)

    def reshape(x, shape, name=None):
        """
        Wrap paddle.reshape
        """
        return manipulation.reshape(x, shape, name)

    def tranpose(x, perm, name=None):
        """
        Wrap paddle.tranpose
        """
        return manipulation.transpose(x, perm, name)

    def concat(x, axis=0, name=None):
        """
        Warp paddle.concat
        """
        return manipulation.concat(x, axis, name)

    def flatten(x, start_axis=0, stop_axis=-1, name=None):
        """
        Warp paddle.flatten
        """
        return manipulation.flatten(x, start_axis, stop_axis, name)
