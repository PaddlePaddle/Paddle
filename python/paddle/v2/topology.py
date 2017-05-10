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


def __flatten__(lis):
    """
    Given a list, possibly nested to any level, return it flattened.
    """
    new_lis = []
    for item in lis:
        if isinstance(item, collections.Sequence):
            new_lis.extend(__flatten__(item))
        else:
            new_lis.append(item)
    return new_lis


def __bfs_travel__(callback, *layers):
    layers = __flatten__(layers)
    for each_layer in layers:
        __break__ = callback(each_layer)
        if __break__:
            return
        __layers__ = each_layer.__parent_layers__.values() + \
                     each_layer.extra_parent()
        __bfs_travel__(callback, *__layers__)


class Topology(object):
    """
    Topology is used to store the information about all layers
    and network configs.
    """

    def __init__(self, layers, extra_layers=None):
        def __check__(layers):
            if not isinstance(layers, collections.Sequence):
                __check_layer_type__(layers)
                layers = [layers]
            for layer in layers:
                __check_layer_type__(layer)
            return layers

        layers = __check__(layers)
        self.layers = layers
        if extra_layers is not None:
            extra_layers = __check__(extra_layers)

        self.__model_config__ = v2_layer.parse_network(
            layers, extra_layers=extra_layers)

        if extra_layers is not None:
            self.layers.extend(extra_layers)

        assert isinstance(self.__model_config__, ModelConfig)

    def proto(self):
        return self.__model_config__

    def get_layer(self, name):
        """
        get v2.Layer Class instance by layer name
        :param name:
        :return:
        """
        result_layer = [None]

        def __impl__(l):
            if l.name == name:
                result_layer[0] = l
                return True  # break
            return False

        __bfs_travel__(__impl__, *self.layers)
        if result_layer[0] is None:
            raise ValueError("No such layer %s" % name)
        return result_layer[0]

    def data_layers(self):
        """
        get all data layer
        :return:
        """
        data_layers = dict()

        def __impl__(l):
            if isinstance(l, v2_layer.DataLayerV2):
                data_layers[l.name] = l

        __bfs_travel__(__impl__, *self.layers)
        return data_layers

    def data_type(self):
        """
        get data_type from proto, such as:
        [('image', dense_vector(768)), ('label', integer_value(10))]
        """
        data_layers = self.data_layers()
        return [(nm, data_layers[nm].type)
                for nm in self.proto().input_layer_names]


def __check_layer_type__(layer):
    if not isinstance(layer, v2_layer.LayerV2):
        raise ValueError('layer should have type paddle.layer.Layer')
