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

from paddle.nn import Layer
from .factory import QuanterFactory

__all__ = ["Stub"]


class Stub(Layer):

    def __init__(self, quanter: QuanterFactory = None):
        super(Stub, self).__init__()
        self._quanter = quanter

    def forward(self, input):
        return input


class QuantStub(Layer):

    def __init__(self, layer: Stub, q_config: dict):
        super(QuantStub, self).__init__()
        self._quanter = None
        if layer._quanter is not None:
            self._quanter = layer._quanter.instance(layer)
        elif "activation" in q_config and q_config["activation"] is not None:
            self._quanter = q_config["activation"].instance(layer)

    def forward(self, input):
        if self._quanter is not None:
            return self._quanter(input)
        else:
            return input
