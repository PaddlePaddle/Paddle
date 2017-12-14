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
import paddle.trainer_config_helpers as conf_helps
import layer as v2_layer
import config_base
import cPickle
from paddle.trainer import config_parser as cp

__all__ = ['Topology']


class Topology(object):
    """
    Topology is used to store the information about all layers
    and network configs.
    """

    def __init__(self, layers, extra_layers=None):
        def __check__(layers):
            if not isinstance(layers, collections.Sequence):
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

    def update_from_default(self):
        # HACK(typhoonzero): update ParameterConfig(proto) in case of
        # optimizers are defined after layers, or between layers.
        # Must be called from trainer.__init__()
        for parameter in self.__model_config__.parameters:
            if parameter.momentum == 0.0 and cp.g_default_momentum:
                parameter.momentum = cp.g_default_momentum
            if parameter.decay_rate == 0.0 and cp.g_default_decay_rate:
                parameter.decay_rate = cp.g_default_decay_rate
            if parameter.initial_mean == 0.0:
                parameter.initial_mean = cp.g_default_initial_mean
            if parameter.initial_std == 0.01:
                parameter.initial_std = cp.g_default_initial_std
            if parameter.initial_strategy == 0:
                parameter.initial_strategy = cp.g_default_initial_strategy
            if parameter.initial_smart == False:
                parameter.initial_smart = cp.g_default_initial_smart
            if parameter.num_batches_regularization == 1 and \
                cp.g_default_num_batches_regularization:
                parameter.num_batches_regularization = \
                    cp.g_default_num_batches_regularization
            if parameter.gradient_clipping_threshold == 0.0 and \
                cp.g_default_gradient_clipping_threshold:
                parameter.gradient_clipping_threshold = \
                    cp.g_default_gradient_clipping_threshold
            if parameter.device == -1 and cp.g_default_device:
                parameter.device = cp.g_default_device
            # FIXME(typhoonzero): ignored: update_hooks, g_default_compact_func

    def use_sparse_updater(self):
        """
        check if any parameter require to use sparse_update
        :return:
        """
        use_sparse = False
        for parameter in self.__model_config__.parameters:
            if parameter.sparse_update or parameter.sparse_remote_update:
                use_sparse = True
                break
        return use_sparse

    def proto(self):
        return self.__model_config__

    def get_layer(self, name):
        """
        get v2.Layer Class instance by layer name
        :param name:
        :return:
        """
        return v2_layer.get_layer(name)

    def data_layers(self):
        """
        get all data layer
        :return:
        """
        data_layers = {}
        for layer in self.proto().layers:
            l = v2_layer.get_layer(layer.name)
            if l and l.layer_type == conf_helps.LayerType.DATA:
                data_layers[layer.name] = l
        return data_layers

    def data_type(self):
        """
        get data_type from proto, such as:
        [('image', dense_vector(768)), ('label', integer_value(10))]
        """
        data_layers = self.data_layers()

        return [(nm, data_layers[nm].data_type)
                for nm in self.proto().input_layer_names]

    def get_layer_proto(self, name):
        for layer in self.__model_config__.layers:
            if layer.name == name:
                return layer
        return None

    def serialize_for_inference(self, stream):
        protobin = self.proto().SerializeToString()
        data_type = self.data_type()
        cPickle.dump({
            'protobin': protobin,
            'data_type': data_type
        }, stream, cPickle.HIGHEST_PROTOCOL)


def __check_layer_type__(layer):
    if not isinstance(layer, config_base.Layer):
        raise ValueError('layer should have type paddle.v2.config_base.Layer')
