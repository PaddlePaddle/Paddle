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
from paddle.nn import functional as F


class QuantedLinear(Layer):
    """
    The computational logic of QuantizedLinear is the same with Linear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self, layer: Layer, q_config):
        super(QuantedLinear, self).__init__()
        # For Linear
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')
        self.name = getattr(layer, 'name')
        # For FakeQuant

        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

    def forward(self, input):
        quant_input = input
        quant_weight = self.weight
        if self.activation_quanter is not None:
            quant_input = self.activation_quanter(input)
        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(self.weight)
        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        out = F.linear(x=input, weight=weight, bias=self.bias, name=self.name)
        return out
