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

import uuid

import paddle.trainer_config_helpers.config_parser_utils as config_parser_utils
import paddle.trainer_config_helpers.layers as layers
import paddle.trainer_config_helpers.networks as networks


class Layer(object):
    def __init__(self, layer=None, is_output=False, **kwargs):
        self.uuid = uuid.uuid4()
        self.parent_layer = layer
        self.layer_output = None
        self.network_config = None
        self.is_output = is_output

    def _execute(self):
        """
        recursively set parent's into proto
        :return:
        """
        if self.parent_layer is not None:
            self.parent_layer.execute()

    def _mark_output_layers(self):
        """
        find out all layers that is marked output and set them into proto
        :return:
        """
        print self.layers()
        output_layers = filter(lambda layer: layer.is_output, self.layers())
        print output_layers
        if len(output_layers) > 0:
            networks.outputs(
                map(lambda layer: layer.layer_output, output_layers))

    def execute(self):
        """
        function to set proto attribute
        :return:
        """
        pass

    def network(self):
        """
        Construct the network according to this layer and all it's parent layers
        :return: return a proto that represent this network.
        """

        def construct_network():
            self.execute()
            self._mark_output_layers()

        if self.network_config is None:
            self.network_config = config_parser_utils.parse_network_config(
                construct_network)
        return self.network_config

    def layers(self):
        """
        get all layers that have relation to this layer.
        :return:
        """
        all_layers = []
        if self.parent_layer is not None:
            all_layers.extend(self.parent_layer.layers())
        all_layers.append(self)
        return all_layers


class DataLayer(Layer):
    def __init__(self,
                 name,
                 size,
                 height=None,
                 width=None,
                 layer_attr=None,
                 **kwargs):
        self.name = name
        self.size = size
        self.height = height
        self.width = width
        self.layer_attr = layer_attr
        super(DataLayer, self).__init__(**kwargs)

    def execute(self):
        self._execute()
        self.layer_output = \
            layers.data_layer(self.name, self.size, self.height, self.width, self.layer_attr)


class FcLayer(Layer):
    def __init__(self,
                 layer,
                 size,
                 act=None,
                 name=None,
                 param_attr=None,
                 bias_attr=None,
                 layer_attr=None,
                 **kwargs):
        self.parent_layer = layer
        self.size = size
        self.act = act
        self.name = name
        self.param_attr = param_attr
        self.bias_attr = bias_attr
        self.layer_attr = layer_attr
        super(FcLayer, self).__init__(layer, **kwargs)

    def execute(self):
        self._execute()
        self.layer_output = \
            layers.fc_layer(input=self.parent_layer.layer_output, size=self.size, act=self.act,
                            name=self.name, param_attr=self.param_attr, bias_attr=self.bias_attr,
                            layer_attr=self.layer_attr)


class ClassificationCost(Layer):
    def __init__(self,
                 layer,
                 label,
                 weight=None,
                 name=None,
                 evaluator=layers.classification_error_evaluator,
                 layer_attr=None,
                 is_output=False,
                 **kwargs):
        assert isinstance(label, Layer)
        self.parent_layer = layer
        self.label = label
        self.weight = weight
        self.name = name
        self.evaluator = evaluator
        self.layer_attr = layer_attr
        super(ClassificationCost, self).__init__(layer, is_output, **kwargs)

    def execute(self):
        self._execute()
        self.label.execute()
        self.layer_output = \
            layers.classification_cost(input=self.parent_layer.layer_output,
                                       label=self.label.layer_output,
                                       weight=self.weight,
                                       name=self.name,
                                       evaluator=self.evaluator,
                                       layer_attr=self.layer_attr)
