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

import copy

from paddle.nn import Layer

from .config import QuantConfig
from .quantize import Quantization


class QAT(Quantization):
    r"""
    Tools used to prepare model for quantization-aware training.
    Args:
        config(QuantConfig) - Quantization configuration

    Examples:
        .. code-block:: python
            from paddle.quantization import QAT, QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
            q_config = QuantConfig(activation=quanter, weight=quanter)
            qat = QAT(q_config)
    """

    def __init__(self, config: QuantConfig):
        super().__init__(config)

    def quantize(self, model: Layer, inplace=False):
        r"""
        Create a model for quantization-aware training.

        The quantization configuration will be propagated in the model.
        And it will insert fake quanters into the model to simulate the quantization.

        Args:
            model(Layer) - The model to be quantized.
            inplace(bool) - Whether to modify the model in-place.

        Return: The prepared model for quantization-aware training.

        Examples:
        .. code-block:: python
            from paddle.quantization import QAT, QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            from paddle.vision.models import LeNet

            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
            q_config = QuantConfig(activation=quanter, weight=quanter)
            qat = QAT(q_config)
            model = LeNet()
            quant_model = qat.quantize(model)
            print(quant_model)
        """
        assert (
            model.training
        ), "Quantization-Aware Training shoud work on training models. Please set training mode by model.train()."
        _model = model if inplace else copy.deepcopy(model)
        self._config._specify(_model)
        self._convert_to_quant_layers(_model, self._config)
        self._insert_activation_observers(_model, self._config)
        return _model
