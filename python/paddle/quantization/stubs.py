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
from .factory import ObserverFactory
from .config import SingleLayerConfig

__all__ = ["Stub"]


class Stub(Layer):
    def __init__(self, observer: ObserverFactory = None):
        super(Stub, self).__init__()
        self._observer = observer

    def forward(self, input):
        return input


class ObserverStub(Layer):
    def __init__(self, layer: Stub, q_config: SingleLayerConfig):
        super(ObserverStub, self).__init__()
        self._observer = None
        if layer._observer is not None:
            self._observer = layer._observer.instance(layer)
        elif q_config.activation is not None:
            self._observer = q_config.activation.instance(layer)

    def forward(self, input):
        if self._observer is not None:
            return self._observer(input)
        else:
            return input
