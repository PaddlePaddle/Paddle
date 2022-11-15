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
from paddle.nn import Layer

__all__ = ["BaseQuanter"]


class BaseQuanter(Layer, metaclass=abc.ABCMeta):
    def __init__(self):
        super(BaseQuanter, self).__init__()

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def scales(self):
        pass

    @abc.abstractmethod
    def zero_points(self):
        pass

    @abc.abstractmethod
    def quant_axis(self):
        pass

    @abc.abstractmethod
    def bit_length(self):
        pass
