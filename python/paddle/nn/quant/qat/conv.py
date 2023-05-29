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
"""
Layers used for QAT.
"""
from paddle.nn import Layer
from paddle.nn import functional as F

from ..format import ConvertibleQuantedLayer


class QuantedConv2D(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedConv2D is the same as Conv2D.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()

        # For Conv2D
        self._groups = layer._groups
        self._stride = layer._stride
        self._padding = layer._padding
        self._padding_mode = layer._padding_mode
        if self._padding_mode != 'zeros':
            self._reversed_padding_repeated_twice = (
                layer._reversed_padding_repeated_twice
            )
        self._dilation = layer._dilation
        self._data_format = layer._data_format
        self.weight = layer.weight
        self.bias = layer.bias

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
        return self._conv_forward(quant_input, quant_weight)

    def _conv_forward(self, inputs, weights):
        if self._padding_mode != 'zeros':
            inputs = F.pad(
                inputs,
                self._reversed_padding_repeated_twice,
                mode=self._padding_mode,
                data_format=self._data_format,
            )
            self._padding = 0

        return F.conv2d(
            inputs,
            weights,
            bias=self.bias,
            padding=self._padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format,
        )

    def weights_to_quanters(self):
        return [('weight', 'weight_quanter')]

    def activation_quanters(self):
        return ['activation_quanter']
