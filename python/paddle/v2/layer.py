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
"""
Before this new package paddle.v2.layer, users would need to use functions
in paddle.trainer_config_helpers.layers to configure networks.

The Old Way:
=========
This old way requires that the creation of a network be defined in a Python
function, say network_config, and that this Python function being passed to
paddle.trainer_config_helpers.parse_network_config for the creation of
protobuf message description of this network.

```python
def network_config():
  img = paddle.trainer_config_helpers.data_layer(name="pixel", size=784)
  inference = paddle.trainer_config_helpers.fc_layer(
    input=img,
    size=10,
    act=paddle.trainer_config_helpers.SoftmaxActivation())
  cost = paddle.trainer_config_helpers.classification_cost(
    input=inference,
    label=paddle.trainer_config_helpers.data_layer(name="label", size=10))

proto_desc = parse_network_config(network_config)
```

When parse_network_config executes network_config, those layer definition
functions like data_layer and fc_layer would change some Python global variables,
so that after the execution, parse_network_config could collect information from
these global variables and generates the protobuf message.



The New Way:
=========
In this PR, we define a function in paddle.v2.layer which creates a Python
class for each layer creation function in paddle.trainer_config_helpers.layers.
Users can use create a network as follows:

```python
img = paddle.v2.layer.data(name="pixel", size=784)
inference = paddle.v2.layer.fc(input=img, size=10, act=paddle.v2.layer.Softmax())
cost = paddle.v2.layer.classification(
  input=inference,
  label=paddle.v2.layer.data(name="label", size=10))

parameters = paddle.v2.parameters.create(cost)
```

This new way doesn't require those invocations to layer definition functions
to be in a Python function but could be anywhere.

Also, the creation of a protobuf message is hidden in the invocation of
paddle.v2.parameters.create, no longer exposed to users.
"""

import collections

import paddle.trainer_config_helpers as conf_helps
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as __parse__
from paddle.trainer_config_helpers.default_decorators import wrap_name_default

import activation
import attr
import data_type

__all__ = [
    'parse_network', 'data', 'fc', 'max_id', 'classification_cost',
    'cross_entropy_cost', 'cross_entropy_with_selfnorm_cost', 'regression_cost',
    'multi_binary_label_cross_entropy_cost', 'rank_cost', 'lambda_cost',
    'sum_cost', 'huber_cost', 'memory', 'embedding', 'recurrent_group'
]


def parse_network(*outputs):
    """
    parse all output layers and then generate a model config proto.
    :param outputs:
    :return:
    """

    def __real_func__():
        context = dict()
        real_output = [each.to_proto(context=context) for each in outputs]
        conf_helps.outputs(real_output)

    return __parse__(__real_func__)


class Layer(object):
    def __init__(self, name, parent_layers, step_input=None):
        assert isinstance(parent_layers, dict)
        assert isinstance(name, basestring)
        self.name = name
        self.step_input = step_input
        self.__parent_layers__ = parent_layers

    def to_proto(self, context):
        """
        function to set proto attribute
        """
        kwargs = dict()
        for layer_name in self.__parent_layers__:
            if not isinstance(self.__parent_layers__[layer_name],
                              collections.Sequence):
                v1_layer = self.__parent_layers__[layer_name].to_proto(
                    context=context)
            else:
                v1_layer = map(lambda x: x.to_proto(context=context),
                               self.__parent_layers__[layer_name])
                if layer_name == "input" and self.step_input is not None:
                    v1_layer.insert(0, self.step_input)
            kwargs[layer_name] = v1_layer

        # memory may have the same name with some layer
        if isinstance(self, MemoryV2):
            return self.to_proto_impl(**kwargs)

        if self.name not in context:
            context[self.name] = self.to_proto_impl(**kwargs)
        return context[self.name]

    def to_proto_impl(self, **kwargs):
        raise NotImplementedError()


def __convert_to_v2__(method_name, name_prefix, parent_names):
    if name_prefix is not None:
        wrapper = wrap_name_default(name_prefix=name_prefix)
    else:
        wrapper = None

    class V2LayerImpl(Layer):
        def __init__(self, name=None, step_input=None, **kwargs):
            parent_layers = dict()
            other_kwargs = dict()
            for pname in parent_names:
                if kwargs.has_key(pname):
                    parent_layers[pname] = kwargs[pname]

            for key in kwargs.keys():
                if key not in parent_names:
                    other_kwargs[key] = kwargs[key]

            super(V2LayerImpl, self).__init__(name, parent_layers, step_input)
            self.__other_kwargs__ = other_kwargs

        if wrapper is not None:
            __init__ = wrapper(__init__)

        def to_proto_impl(self, **kwargs):
            args = dict()
            for each in kwargs:
                args[each] = kwargs[each]
            for each in self.__other_kwargs__:
                args[each] = self.__other_kwargs__[each]
            return getattr(conf_helps, method_name)(name=self.name, **args)

    return V2LayerImpl


"""
Some layer may need some special config, and can not use __convert_to_v2__ to convert.
So we also need to implement some special LayerV2.
"""


class DataLayerV2(Layer):
    def __init__(self, name, type, **kwargs):
        assert isinstance(type, data_type.InputType)

        self.type = type
        self.__method_name__ = 'data_layer'
        self.__kwargs__ = kwargs

        super(DataLayerV2, self).__init__(name=name, parent_layers=dict())

    def to_proto_impl(self, **kwargs):
        args = dict()
        args['size'] = self.type.dim
        for each in kwargs:
            args[each] = kwargs[each]
        for each in self.__kwargs__:
            args[each] = self.__kwargs__[each]
        return getattr(conf_helps, self.__method_name__)(name=self.name, **args)


class MemoryV2(Layer):
    def __init__(self, name, size, **kwargs):
        self.name = name
        self.size = size
        self.__kwargs__ = kwargs
        super(MemoryV2, self).__init__(name=name, parent_layers=dict())

    def to_proto_impl(self, **kwargs):
        args = dict()
        for each in kwargs:
            args[each] = kwargs[each]
        for each in self.__kwargs__:
            args[each] = self.__kwargs__[each]
        return conf_helps.memory(name=self.name, size=self.size, **args)


data = DataLayerV2
fc = __convert_to_v2__('fc_layer', name_prefix='fc', parent_names=['input'])
max_id = __convert_to_v2__(
    'maxid_layer', name_prefix='maxid', parent_names=['input'])
classification_cost = __convert_to_v2__(
    'classification_cost',
    name_prefix='classification_cost',
    parent_names=['input', 'label', 'weight'])
regression_cost = __convert_to_v2__(
    'regression_cost',
    name_prefix='regression_cost',
    parent_names=['input', 'label', 'weight'])
cross_entropy_cost = __convert_to_v2__(
    'cross_entropy',
    name_prefix='cross_entropy',
    parent_names=['input', 'label'])
embedding = __convert_to_v2__(
    'embedding_layer', name_prefix='embedding', parent_names=['input'])
last_seq = __convert_to_v2__(
    'last_seq', name_prefix='last_seq', parent_names=['input'])
recurrent_group = __convert_to_v2__(
    'recurrent_group', name_prefix='recurrent_layer', parent_names=['input'])
memory = MemoryV2

cross_entropy_with_selfnorm_cost = __convert_to_v2__(
    'cross_entropy_with_selfnorm',
    name_prefix='cross_entropy_with_selfnorm',
    parent_names=['input', 'label'])
multi_binary_label_cross_entropy_cost = __convert_to_v2__(
    'multi_binary_label_cross_entropy',
    name_prefix='multi_binary_label_cross_entropy',
    parent_names=['input', 'label'])
rank_cost = __convert_to_v2__(
    'rank_cost',
    name_prefix='rank_cost',
    parent_names=['left', 'right', 'label', 'weight'])
lambda_cost = __convert_to_v2__(
    'lambda_cost', name_prefix='lambda_cost', parent_names=['input', 'score'])
sum_cost = __convert_to_v2__(
    'sum_cost', name_prefix='sum_cost', parent_names=['input'])
huber_cost = __convert_to_v2__(
    'huber_cost', name_prefix='huber_cost', parent_names=['input', 'label'])

if __name__ == '__main__':
    pixel = data(name='pixel', type=data_type.dense_vector(784))
    label = data(name='label', type=data_type.integer_value(10))
    weight = data(name='weight', type=data_type.dense_vector(10))
    score = data(name='score', type=data_type.dense_vector(1))

    hidden = fc(input=pixel,
                size=100,
                act=activation.Sigmoid(),
                param_attr=attr.Param(name='hidden'))
    inference = fc(input=hidden, size=10, act=activation.Softmax())
    maxid = max_id(input=inference)
    cost1 = classification_cost(input=inference, label=label)
    cost2 = classification_cost(input=inference, label=label, weight=weight)
    cost3 = cross_entropy_cost(input=inference, label=label)
    cost4 = cross_entropy_with_selfnorm_cost(input=inference, label=label)
    cost5 = regression_cost(input=inference, label=label)
    cost6 = regression_cost(input=inference, label=label, weight=weight)
    cost7 = multi_binary_label_cross_entropy_cost(input=inference, label=label)
    cost8 = rank_cost(left=score, right=score, label=score)
    cost9 = lambda_cost(input=inference, score=score)
    cost10 = sum_cost(input=inference)
    cost11 = huber_cost(input=score, label=label)

    mem = memory(name="rnn_state", size=10)

    # print parse_network(cost1)
    # print parse_network(cost2)
    # print parse_network(cost1, cost2)
    # print parse_network(cost2)
    # print parse_network(inference, maxid)
    print parse_network(cost1, cost2)
    print parse_network(cost3, cost4)
    print parse_network(cost5, cost6)
    print parse_network(cost7, cost8, cost9, cost10, cost11)
    print parse_network(inference, maxid)
