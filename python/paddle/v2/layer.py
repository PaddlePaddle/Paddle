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
import inspect

import paddle.trainer_config_helpers as conf_helps
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as __parse__

from paddle.trainer_config_helpers.default_decorators import wrap_name_default
from paddle.trainer_config_helpers.default_decorators import wrap_act_default
from paddle.trainer_config_helpers.default_decorators import \
    wrap_bias_attr_default
from paddle.trainer_config_helpers.layers import layer_support

import data_type
import activation

__all__ = ['parse_network', 'data']

__projection_names__ = filter(lambda x: x.endswith('_projection'),
                              dir(conf_helps))
__all__ += __projection_names__

__operator_names__ = filter(lambda x: x.endswith('_operator'), dir(conf_helps))
__all__ += __operator_names__


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
    def __init__(self, name=None, parent_layers=None):
        assert isinstance(parent_layers, dict)
        self.name = name
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
            kwargs[layer_name] = v1_layer

        if self.name is None:
            return self.to_proto_impl(**kwargs)
        elif self.name not in context:
            context[self.name] = self.to_proto_impl(**kwargs)

        return context[self.name]

    def to_proto_impl(self, **kwargs):
        raise NotImplementedError()


def __convert_to_v2__(method_name, parent_names, is_default_name=True):
    if is_default_name:
        wrapper = wrap_name_default(name_prefix=method_name)
    else:
        wrapper = None

    class V2LayerImpl(Layer):
        def __init__(self, **kwargs):
            parent_layers = dict()
            other_kwargs = dict()
            for pname in parent_names:
                if kwargs.has_key(pname):
                    parent_layers[pname] = kwargs[pname]

            for key in kwargs.keys():
                if key not in parent_names:
                    other_kwargs[key] = kwargs[key]

            name = kwargs.get('name', None)
            super(V2LayerImpl, self).__init__(name, parent_layers)
            self.__other_kwargs__ = other_kwargs

        if wrapper is not None:
            __init__ = wrapper(__init__)

        def to_proto_impl(self, **kwargs):
            args = dict()
            for each in kwargs:
                args[each] = kwargs[each]
            for each in self.__other_kwargs__:
                args[each] = self.__other_kwargs__[each]
            return getattr(conf_helps, method_name)(**args)

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


class MixedLayerV2(Layer):
    """
    This class is use to support `with` grammar. If not, the following code
    could convert mixed_layer simply.

        mixed = __convert_to_v2__(
            'mixed_layer', name_prefix='mixed', parent_names=['input'])
    """

    class AddToSealedMixedLayerExceptionV2(Exception):
        pass

    def __init__(self,
                 size=0,
                 input=None,
                 name=None,
                 act=None,
                 bias_attr=None,
                 layer_attr=None):
        self.__method_name__ = 'mixed_layer'
        self.finalized = False
        self.__inputs__ = []
        if input is not None:
            self.__inputs__ = input

        other_kwargs = dict()
        other_kwargs['name'] = name
        other_kwargs['size'] = size
        other_kwargs['act'] = act
        other_kwargs['bias_attr'] = bias_attr
        other_kwargs['layer_attr'] = layer_attr

        parent_layers = {"input": self.__inputs__}
        super(MixedLayerV2, self).__init__(name, parent_layers)
        self.__other_kwargs__ = other_kwargs

    def __iadd__(self, other):
        if not self.finalized:
            self.__inputs__.append(other)
            return self
        else:
            raise MixedLayerTypeV2.AddToSealedMixedLayerExceptionV2()

    def __enter__(self):
        assert len(self.__inputs__) == 0
        return self

    def __exit__(self, *args, **kwargs):
        self.finalized = True

    def to_proto_impl(self, **kwargs):
        args = dict()
        for each in kwargs:
            args[each] = kwargs[each]
        for each in self.__other_kwargs__:
            args[each] = self.__other_kwargs__[each]
        return getattr(conf_helps, self.__method_name__)(**args)


@wrap_name_default("mixed")
@wrap_act_default(act=activation.Linear())
@wrap_bias_attr_default(has_bias=False)
@layer_support(conf_helps.layers.ERROR_CLIPPING, conf_helps.layers.DROPOUT)
def mixed(size=0,
          name=None,
          input=None,
          act=None,
          bias_attr=False,
          layer_attr=None):
    return MixedLayerV2(size, input, name, act, bias_attr, layer_attr)


LayerV2 = Layer
data = DataLayerV2
AggregateLevel = conf_helps.layers.AggregateLevel
ExpandLevel = conf_helps.layers.ExpandLevel


def __layer_name_mapping__(inname):
    if inname in ['data_layer', 'memory', 'mixed_layer']:
        # Do Not handle these layers
        return
    elif inname == 'maxid_layer':
        return 'max_id'
    elif inname.endswith('memory') or inname.endswith(
            '_seq') or inname.endswith('_sim') or inname == 'hsigmoid':
        return inname
    elif inname in [
            'cross_entropy', 'multi_binary_label_cross_entropy',
            'cross_entropy_with_selfnorm'
    ]:
        return inname + "_cost"
    elif inname.endswith('_cost'):
        return inname
    elif inname.endswith("_layer"):
        return inname[:-len("_layer")]


def __layer_name_mapping_parent_names__(inname):
    all_args = getattr(conf_helps, inname).argspec.args
    return filter(
        lambda x: x in ['input1', 'input2','label', 'input', 'a', 'b', 'expand_as',
                        'weights', 'vectors', 'weight', 'score', 'left', 'right'],
        all_args)


def __convert_layer__(_new_name_, _old_name_, _parent_names_):
    global __all__
    __all__.append(_new_name_)
    globals()[new_name] = __convert_to_v2__(_old_name_, _parent_names_)


for each_layer_name in dir(conf_helps):
    new_name = __layer_name_mapping__(each_layer_name)
    if new_name is not None:
        parent_names = __layer_name_mapping_parent_names__(each_layer_name)
        assert len(parent_names) != 0, each_layer_name
        __convert_layer__(new_name, each_layer_name, parent_names)

del parent_names
del new_name
del each_layer_name

# convert projection
for prj in __projection_names__:
    globals()[prj] = __convert_to_v2__(
        prj, parent_names=['input'], is_default_name=False)

# convert operator
operator_list = [
    # [V1_method_name, parent_names],
    ['dotmul_operator', ['a', 'b']],
    ['conv_operator', ['img', 'filter']]
]
for op in operator_list:
    globals()[op[0]] = __convert_to_v2__(
        op[0], parent_names=op[1], is_default_name=False)
