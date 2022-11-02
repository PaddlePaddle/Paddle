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

import paddle
import paddle.nn as nn
from paddle.nn import Layer
from typing import Dict
from .wrapper import ObserveWrapper
from .stubs import ObserverStub, Stub
from .factory import ObserverFactory
from typing import Union

__all__ = ["QuantConfig", "TRTQuantConfig"]

DEFAULT_QAT_LAYER_MAPPINGS: Dict[Layer, Layer] = {
    Stub: ObserverStub,
    nn.Conv2D: nn.quant.qat.QuantConv2D,
    nn.Conv2DTranspose: nn.quant.quant_layers.QuantizedConv2DTranspose,
    nn.Linear: nn.quant.qat.QuantLinear,
}

DEFAULT_LEAVES = [nn.ReLU, nn.AvgPool2D]


class SingleLayerConfig(object):
    def __init__(self, activation: ObserverFactory, weight: ObserverFactory):
        self._activation = activation
        self._weight = weight

    @property
    def activation(self):
        return self._activation

    @property
    def weight(self):
        return self._weight


class QuantConfig(object):
    def __init__(self, activation: ObserverFactory, weight: ObserverFactory):
        self._global_config = SingleLayerConfig(activation, weight)
        self._layer2config = {}
        self._prefix2config = {}
        self._type2config = {}
        self._model = None
        self._qat_layer_mapping = DEFAULT_QAT_LAYER_MAPPINGS
        self._costum_leaves = []

    def add_group(
        self,
        group: Union[type, str, list],
        activation: ObserverFactory = None,
        weight: ObserverFactory = None,
    ):
        config = SingleLayerConfig(activation, weight)
        if isinstance(group, type) and issubclass(group, paddle.nn.Layer):
            self._type2config[group] = config
        if isinstance(group, str):
            self._prefix2config[group] = config
        if isinstance(group, list):
            for _element in group:
                self.add_group(_element, activation=activation, weight=weight)

    def add_qat_layer_mapping(self, source: Layer, target: Layer):
        assert isinstance(source, type) and issubclass(
            source, paddle.nn.Layer
        ), "The source layer to be placed should be a subclass of paddle.nn.Layer"
        assert isinstance(target, type) and issubclass(
            source, paddle.nn.Layer
        ), "The target layer should be a subclass of paddle.nn.qat.Layer"
        self._qat_layer_mapping[source] = target

    def add_costum_leaf(self, layer: Layer):
        self._costum_leaves.append(layer)

    @property
    def costum_leaves(self):
        return self._costum_leaves

    def get_qat_layer(self, layer: Layer):
        q_config = self.get_config_by_layer(layer)
        return self.qat_layer_mappings[type(layer)](layer, q_config)

    def need_observe(self, layer: Layer):
        return self.is_leaf(layer) and self.has_observer_config(layer)

    def has_observer_config(self, layer: Layer):
        _config = self.get_config_by_layer(layer)
        return _config is not None and _config.activation is not None

    def is_leaf(self, layer: Layer):
        return (
            self.is_default_leaf(layer)
            or self.is_real_leaf(layer)
            or self.is_custom_leaf(layer)
        )

    def is_default_leaf(self, layer: Layer):
        return layer in DEFAULT_LEAVES

    def is_real_leaf(self, layer: Layer):
        return layer._sub_layers is None or len(layer._sub_layers) == 0

    def is_custom_leaf(self, layer: Layer):
        return layer in self.costum_leaves

    def get_observer(self, layer):
        _config = self.get_config_by_layer(layer)
        _observer = None if _config is None else _config.activation
        return None if _observer is None else _observer.instance(layer)

    def get_observe_wrapper(self, layer: Layer):
        _observer = self.get_observer(layer)
        return ObserveWrapper(_observer, layer)

    @property
    def qat_layer_mappings(self):
        return self._qat_layer_mapping

    @property
    def default_qat_layer_mapping(self):
        return DEFAULT_QAT_LAYER_MAPPINGS

    @property
    def global_config(self) -> SingleLayerConfig:
        return self._global_config

    def get_config_by_layer(self, layer) -> SingleLayerConfig:
        return self._layer2config.get(layer, None)

    def is_quantable(self, layer: Layer):
        return layer in self._layer2config

    def specify(self, model: Layer):
        self._model = model
        self.specify_helper(self._model)

    def specify_helper(self, model: Layer, prefix: str = ""):
        for name, child in model.named_children():
            layer_prefix = "/".join([prefix, name])
            config = self._layer2config.get(model, self.global_config)
            config = self._type2config.get(type(child), config)
            self._layer2config[child] = self._prefix2config.get(
                layer_prefix, config
            )
            self.specify_helper(child, prefix=layer_prefix)
        return self

    def details(self):
        return self.details_helper(self._model)

    def details_helper(self, layer: Layer):
        extra_lines = []
        sublayer_lines = []
        for name, sublayer in layer.named_children():
            sublayer_str = self.details_helper(sublayer)
            sublayer_str = self._addindent(sublayer_str, 2)
            sublayer_lines.append(
                '('
                + name
                + '): '
                + sublayer_str
                + ', '
                + str(self._layer2config[sublayer])
            )

        final_str = layer.__class__.__name__ + '('
        if extra_lines:
            if len(extra_lines) > 1:
                final_str += '\n  ' + '\n  '.join(extra_lines) + '\n'
            elif len(extra_lines) == 1:
                final_str += extra_lines[0]
        if sublayer_lines:
            final_str += '\n  ' + '\n  '.join(sublayer_lines) + '\n'

        final_str += ')'
        return final_str

    def _addindent(self, string, indent):
        s1 = string.split('\n')
        if len(s1) == 1:
            return string
        s2 = []
        for idx, line in enumerate(s1):
            if idx > 0:
                s2.append(str((indent * ' ') + line))
        return s1[0] + '\n' + '\n'.join(s2)

    def __str__(self):
        result = ""
        result += f"Global config:\nactivation: {self._activation}\nweight: {self._weight}\n"
        if len(self._type2config) > 0:
            result += f"Layer type config:\n{self._type2config}\n"
        if len(self._prefix2config) > 0:
            result += f"Layer prefix config: \n{self._prefix2config}\n"
        return result

    # def from_dict(self, config):

    # def


class TRTQuantConfig(QuantConfig):
    def __init__(self, activation: ObserverFactory, weight: ObserverFactory):
        super(TRTQuantConfig, self).__init__(activation, weight)
