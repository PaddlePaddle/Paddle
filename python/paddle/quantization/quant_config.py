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
from typing import Dict, Any

__all__ = ["QuantConfig", "TRTQuantConfig"]

DEFAULT_QAT_LAYER_MAPPINGS: Dict[paddle.nn.Layer, paddle.nn.Layer] = {
    paddle.nn.Conv2D: paddle.nn.quant.qat.QuantConv2D,
    paddle.nn.Conv2DTranspose:
    paddle.nn.quant.quant_layers.QuantizedConv2DTranspose,
    paddle.nn.Linear: paddle.nn.quant.qat.QuantLinear,
}


class QuantConfig(object):

    def __init__(self, activation, weight):
        self._activation = activation
        self._weight = weight
        self._layer2config = {}
        self._prefix2config = {}
        self._type2config = {}
        self._model = None
        self._qat_layer_mapping = DEFAULT_QAT_LAYER_MAPPINGS

    def add_group(self, group, activation=None, weight=None):
        config = {"weight": weight, "activation": activation}
        if isinstance(group, type) and issubclass(group, paddle.nn.Layer):
            self._type2config[group] = config
        if isinstance(group, str):
            self._prefix2config[group] = config
        if isinstance(group, list):
            for _element in group:
                self.add_group(_element, activation=activation, weight=weight)

    def add_qat_layer_mapping(self, source, target):
        assert isinstance(source, type) and issubclass(
            source, paddle.nn.Layer
        ), "The source layer to be placed should be a subclass of paddle.nn.Layer"
        assert isinstance(target, type) and issubclass(
            source, paddle.nn.Layer
        ), "The target layer should be a subclass of paddle.nn.qat.Layer"
        self._qat_layer_mapping[source] = target

    def get_qat_layer(self, layer):
        q_config = self.get_config_by_layer(layer)
        return self.qat_layer_mappings[type(layer)](layer, q_config)

    @property
    def qat_layer_mappings(self):
        return self._qat_layer_mapping

    @property
    def default_qat_layer_mapping(self):
        return DEFAULT_QAT_LAYER_MAPPINGS

    @property
    def global_config(self):
        return {"weight": self._weight, "activation": self._activation}

    def get_config_by_layer(self, layer):
        return self._layer2config.get(layer, None)

    def is_quantable(self, layer):
        return layer in self._layer2config

    def specify(self, model):
        self._model = model
        self.specify_helper(self._model)

    def specify_helper(self, model, prefix=""):
        for name, child in model.named_children():
            layer_prefix = "/".join([prefix, name])
            config = self._layer2config.get(model, self.global_config)
            config = self._type2config.get(type(child), config)
            self._layer2config[child] = self._prefix2config.get(
                layer_prefix, config)
            self.specify_helper(child, prefix=layer_prefix)
        return self

    def details(self):
        return self.details_helper(self._model)

    def details_helper(self, layer):
        extra_lines = []
        sublayer_lines = []
        for name, sublayer in layer.named_children():
            sublayer_str = self.details_helper(sublayer)
            sublayer_str = self._addindent(sublayer_str, 2)
            sublayer_lines.append('(' + name + '): ' + sublayer_str + ', ' +
                                  str(self._layer2config[sublayer]))

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

    def __init__(self, activation, weight):
        super(TRTQuantConfig, self).__init__(activation, weight)
