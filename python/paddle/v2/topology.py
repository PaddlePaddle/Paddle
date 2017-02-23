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

import paddle.trainer_config_helpers as conf_helps
from paddle.proto.ModelConfig_pb2 import ModelConfig

import data_type
import layer as v2_layer

__all__ = ['Topology']


class Topology(object):
    """
    Topology is used to store the information about all layers
    and network configs.
    """

    def __init__(self, layers):
        if not isinstance(layers, collections.Sequence):
            raise ValueError("input of Topology should be a list of Layer")
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

    def get_data_layer(self):
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

    def get_layer_proto(self, name):
        """
        get layer by layer name
        :param name:
        :return:
        """
        layers = filter(lambda layer: layer.name == name,
                        self.__model_config__.layers)
        if len(layers) is 1:
            return layers[0]
        else:
            return None

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


if __name__ == '__main__':
    pixel = v2_layer.data(name='pixel', type=data_type.dense_vector(784))
    label = v2_layer.data(name='label', type=data_type.integer_value(10))
    hidden = v2_layer.fc(input=pixel,
                         size=100,
                         act=conf_helps.SigmoidActivation())
    inference = v2_layer.fc(input=hidden,
                            size=10,
                            act=conf_helps.SoftmaxActivation())
    maxid = v2_layer.max_id(input=inference)
    cost1 = v2_layer.classification_cost(input=inference, label=label)
    cost2 = v2_layer.cross_entropy_cost(input=inference, label=label)

    print Topology(cost1).proto()
    print Topology(cost2).proto()
    print Topology(cost1, cost2).proto()
    print Topology(cost2).proto()
    print Topology(inference, maxid).proto()
