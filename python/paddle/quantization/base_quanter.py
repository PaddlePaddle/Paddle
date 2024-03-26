# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import abc
from collections.abc import Iterable
from typing import Union

import numpy as np

import paddle
from paddle.nn import Layer


class BaseQuanter(Layer, metaclass=abc.ABCMeta):
    r"""
    Built-in quanters and customized quanters should extend this base quanter
    and implement abstract methods.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def scales(self) -> Union[paddle.Tensor, np.ndarray]:
        r"""
        Get the scales used for quantization.
        It can be none which means the quanter didn't hold scales for quantization.
        """
        pass

    @abc.abstractmethod
    def zero_points(self) -> Union[paddle.Tensor, np.ndarray]:
        r"""
        Get the zero points used for quantization.
        It can be none which means the quanter didn't hold zero points for quantization.
        """
        pass

    @abc.abstractmethod
    def quant_axis(self) -> Union[int, Iterable]:
        r"""
        Get the axis of quantization. None means tensor-wise quantization.
        """
        pass

    @abc.abstractmethod
    def bit_length(self):
        r"""
        Get the bit length of quantization.
        """
        pass
