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
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from paddle.nn import Layer

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from paddle import Tensor


class BaseQuanter(Layer, metaclass=abc.ABCMeta):
    r"""
    Built-in quanters and customized quanters should extend this base quanter
    and implement abstract methods.
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, input: Tensor) -> Tensor | npt.NDArray[Any]:
        pass

    @abc.abstractmethod
    def scales(self) -> Tensor | npt.NDArray[Any]:
        r"""
        Get the scales used for quantization.
        It can be none which means the quanter didn't hold scales for quantization.
        """
        pass

    @abc.abstractmethod
    def zero_points(self) -> Tensor | npt.NDArray[Any]:
        r"""
        Get the zero points used for quantization.
        It can be none which means the quanter didn't hold zero points for quantization.
        """
        pass

    @abc.abstractmethod
    def quant_axis(self) -> int | Iterable[int]:
        r"""
        Get the axis of quantization. None means tensor-wise quantization.
        """
        pass

    @abc.abstractmethod
    def bit_length(self) -> int | Iterable[int]:
        r"""
        Get the bit length of quantization.
        """
        pass
