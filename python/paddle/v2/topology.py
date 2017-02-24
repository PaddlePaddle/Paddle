# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import collections

from paddle.proto.ModelConfig_pb2 import ModelConfig

import layer as v2_layer

__all__ = ['Topology']


class Topology(object):
    """
    Topology is used to store the information about all layers
    and network configs.
    """

    def __init__(self, layers):
        if not isinstance(layers, collections.Sequence):
            __check_layer_type__(layers)
            layers = [layers]
        for layer in layers:
            if not isinstance(layer, v2_layer.LayerV2):
                raise ValueError('layer should have type paddle.layer.Layer')
        self.layers = layers
        self.__model_config__ = v2_layer.parse_network(*layers)
        assert isinstance(self.__model_config__, ModelConfig)

    def proto(self):
        return self.__model_config__

    def get_layer(self, name):
        """
        get v2.Layer Class instance by layer name
        :param name:
        :return:
        """
        result_layer = []

        def find_layer_by_name(layer, layer_name):
            if layer.name == layer_name and len(result_layer) == 0:
                result_layer.append(layer)
            for parent_layer in layer.__parent_layers__.values():
                find_layer_by_name(parent_layer, layer_name)

        for layer in self.layers:
            find_layer_by_name(layer, name)

        return result_layer[0]

    def data_layer(self):
        """
        get all data layer
        :return:
        """
        data_layers = []

        def find_data_layer(layer):
            assert isinstance(layer, layer.LayerV2)
            if isinstance(layer, v2_layer.DataLayerV2):
                if len(
                        filter(lambda data_layer: data_layer.name == layer.name,
                               data_layers)) == 0:
                    data_layers.append(layer)
            for parent_layer in layer.__parent_layers__.values():
                find_data_layer(parent_layer)

        for layer in self.layers:
            find_data_layer(layer)

        return data_layers

    def data_type(self):
        """
        get data_type from proto, such as:
        [('image', dense_vector(768)), ('label', integer_value(10))]
        the order is the same with __model_config__.input_layer_names
        """
        data_types_lists = []
        for layer_name in self.__model_config__.input_layer_names:
            data_types_lists.append(
                (layer_name, self.get_layer(layer_name).type))

        return data_types_lists


def __check_layer_type__(layer):
    if not isinstance(layer, v2_layer.LayerV2):
        raise ValueError('layer should have type paddle.layer.Layer')
