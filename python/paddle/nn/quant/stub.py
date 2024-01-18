# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

""" Define stub used in quantization."""

from ..layer.layers import Layer


class Stub(Layer):
    r"""
    The stub is used as placeholders that will be replaced by observers before PTQ or QAT.
    It is hard to assign a quantization configuration to a functional API called in
    the forward of a layer. Instead, we can create a stub and add it to the sublayers of the layer.
    And call the stub before the functional API in the forward. The observer held by the
    stub will observe or quantize the inputs of the functional API.

    Args:
        observer(QuanterFactory): The configured information of the observer to be inserted.
            It will use a global configuration to create the observers if the 'observer' is none.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.nn.quant import Stub
            >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            >>> from paddle.nn import Conv2D
            >>> from paddle.quantization import QAT, QuantConfig

            >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
            >>> class Model(paddle.nn.Layer):
            ...     def __init__(self, num_classes=10):
            ...         super().__init__()
            ...         self.conv = Conv2D(3, 6, 3, stride=1, padding=1)
            ...         self.quant = Stub(quanter)
            ...
            ...     def forward(self, inputs):
            ...         out = self.conv(inputs)
            ...         out = self.quant(out)
            ...         return paddle.nn.functional.relu(out)

            >>> model = Model()
            >>> q_config = QuantConfig(activation=quanter, weight=quanter)
            >>> qat = QAT(q_config)
            >>> quant_model = qat.quantize(model)
            >>> print(quant_model)
            Model(
                (conv): QuantedConv2D(
                    (weight_quanter): FakeQuanterWithAbsMaxObserverLayer()
                    (activation_quanter): FakeQuanterWithAbsMaxObserverLayer()
                )
                (quant): QuanterStub(
                    (_observer): FakeQuanterWithAbsMaxObserverLayer()
                )
            )
    """

    def __init__(self, observer=None):
        super().__init__()
        self._observer = observer

    def forward(self, input):
        return input


class QuanterStub(Layer):
    r"""
    It is an identity layer with an observer observing the input.
    Before QAT or PTQ, the stub in the model will be replaced with an instance of QuanterStub.
    The user should not use this class directly.

    Args:
        layer(paddle.nn.Layer): The stub layer with an observer configure factory. If the observer
        of the stub layer is none, it will use 'q_config' to create an observer instance.
        q_config(QuantConfig): The quantization configuration for the current stub layer.
    """

    def __init__(self, layer: Stub, q_config):
        super().__init__()
        self._observer = None
        if layer._observer is not None:
            self._observer = layer._observer._instance(layer)
        elif q_config.activation is not None:
            self._observer = q_config.activation._instance(layer)

    def forward(self, input):
        return self._observer(input) if self._observer is not None else input
