"""Abstract observer class."""

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

from paddle.nn.layer.layers import LayerABCMeta

from .base_quanter import BaseQuanter


class BaseObserver(BaseQuanter, metaclass=LayerABCMeta):
    r"""
    Built-in observers and customized observers should extend this base observer
    and implement abstract methods.
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def cal_thresholds(self) -> None:
        pass
