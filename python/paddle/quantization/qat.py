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

__all__ = ["QAT"]


class QAT(object):
    """
    Applying quantization-aware training (QAT) to the model.
    """

    def __init__(self, config: QuantConfig):
        self._config = copy.deepcopy(config)

    def quantize(self, model: Layer, inplace=False):
        _model = model if inplace else copy.deepcopy(model)
        self._config.specify(_model)
        self._convert_to_quant_layers(_model, self._config)
        self._insert_activation_observers(_model, self._config)
        return _model

    def _convert_to_quant_layers(self, model: Layer, config: QuantConfig):
        replaced = {}
        for name, child in model.named_children():
            if config.is_quantable(child):
                if type(child) not in config.qat_layer_mappings:
                    self._convert_to_quant_layers(child, config)
                else:
                    replaced[name] = config.get_qat_layer(child)
        for key, value in replaced.items():
            model._sub_layers[key] = value

    def _insert_activation_observers(self, model: Layer, config: QuantConfig):
        replaced = {}
        for name, child in model.named_children():
            if config.need_observe(child):
                replaced[name] = config.get_observe_wrapper(child)
            else:
                self._insert_activation_observers(child, config)
        for key, value in replaced.items():
            model._sub_layers[key] = value

    def details(self):
        return self._config.details()

    def __str__(self):
        return self.details()

    def __repr__(self):
        return self.__str__()

    # def convert(self, model, inplce=False):
