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
from config_base import Layer, __convert_to_v2__
import paddle.trainer_config_helpers as conf_helps
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as __parse__
from paddle.trainer_config_helpers.default_decorators import wrap_act_default
from paddle.trainer_config_helpers.default_decorators import \
    wrap_bias_attr_default
from paddle.trainer_config_helpers.default_decorators import wrap_name_default
from paddle.trainer_config_helpers.layers import layer_support
from paddle.trainer.config_parser import \
    RecurrentLayerGroupWithoutOutLinksBegin, RecurrentLayerGroupSetOutLink, \
    RecurrentLayerGroupEnd, model_type

import activation
import data_type

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


class WithExtraParent(Layer):
    def extra_parent(self):
        return self.__extra_parent__

    def __init__(self, name=None, parent_layers=None):
        self.__extra_parent__ = []
        super(WithExtraParent, self).__init__(
            name=name, parent_layers=parent_layers)

    def append_extra_parent(self, parent):
        self.__extra_parent__.append(parent)

    def to_proto(self, context):
        """
        function to set proto attribute
        """
        kwargs = dict()
        for p in self.__extra_parent__:
            p.to_proto(context=context)

        for layer_name in self.__parent_layers__:
            if not isinstance(self.__parent_layers__[layer_name],
                              collections.Sequence):
                v1_layer = self.__parent_layers__[layer_name].to_proto(
                    context=context)
            else:
                v1_layer = map(lambda x: x.to_proto(context=context),
                               self.__parent_layers__[layer_name])
            kwargs[layer_name] = v1_layer

        if self.context_name() is None:
            return self.to_proto_impl(context=context, **kwargs)
        elif self.context_name() not in context:
            context[self.context_name()] = self.to_proto_impl(
                context=context, **kwargs)

        if self.use_context_name():
            return context[self.context_name()]
        else:
            return context[self.name]


class MemoryV2(WithExtraParent):
    def __init__(self, name, **kwargs):
        self.name = name
        super(MemoryV2, self).__init__(name=name, parent_layers=dict())
        self.__kwargs__ = kwargs
        self.__boot_layer_name__ = None
        if 'boot_layer' in kwargs:
            begin_of_current_rnn = []
            # TODO(yuyang18): Fix inspect, it could be wrong when user invoke a
            # function inside step.
            st = inspect.stack()
            for i in xrange(len(st)):
                locs = inspect.stack()[i][0].f_locals
                keys = locs.keys()
                for key in keys:
                    val = locs[key]
                    if isinstance(val, RecurrentLayerInput):
                        begin_of_current_rnn.append(val)
                    elif isinstance(val, collections.Sequence):
                        for v in val:
                            if isinstance(v, RecurrentLayerInput):
                                begin_of_current_rnn.append(v)

                if begin_of_current_rnn:
                    break
            assert begin_of_current_rnn is not None
            for extra in begin_of_current_rnn:
                self.append_extra_parent(extra)
                assert isinstance(extra, WithExtraParent)
                extra.append_extra_parent(kwargs['boot_layer'])
                self.__boot_layer_name__ = kwargs['boot_layer'].name

    def to_proto_impl(self, context, **kwargs):
        args = dict()
        for each in kwargs:
            args[each] = kwargs[each]
        for each in self.__kwargs__:
            args[each] = self.__kwargs__[each]

        if self.__boot_layer_name__ is not None:
            args['boot_layer'] = context[self.__boot_layer_name__]

        size = args.get('size', None)
        if size is not None:
            if callable(size):
                real_size = size()
            else:
                real_size = size
            args['size'] = real_size
        return conf_helps.memory(name=self.name, **args)

    def context_name(self):
        return self.name + "#memory"

    def use_context_name(self):
        """
        memory layer will have the same name with some layer
        :return:
        """
        return True


class LayerOutputV2(Layer):
    """
    LayerOutputV2 is used to store the result of LayerOutput in v1 api.
    It will not store it's parents because layer_output has been parsed already.
    """

    def __init__(self, layer_output):
        assert isinstance(layer_output, conf_helps.LayerOutput)
        self.layer_output = layer_output
        super(LayerOutputV2, self).__init__(
            name=layer_output.name, parent_layers=dict())

    def to_proto_impl(self):
        return self.layer_output


class StaticInputV2(object):
    def __init__(self, input, is_seq=False, size=None):
        assert isinstance(input, LayerV2)
        self.name = input.name
        self.input = input
        self.is_seq = is_seq
        self.size = size
        # TODO(qiaolongfei): add size
        # assert input.size is not None or size is not None


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
            raise MixedLayerV2.AddToSealedMixedLayerExceptionV2()

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
        size = args.get('size', None)
        if size is not None:
            if callable(size):
                real_size = size()
            else:
                real_size = size
            args['size'] = real_size
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


class RecurrentLayerInput(WithExtraParent):
    def __init__(self, recurrent_name, index, parent_layers):
        assert len(parent_layers) == 1
        self.__parents__ = parent_layers.values()[0]
        super(RecurrentLayerInput, self).__init__(
            name=self.__parents__[index].name, parent_layers=parent_layers)
        self.__recurrent_name__ = recurrent_name

    def context_name(self):
        return self.__recurrent_name__ + ".begin"

    def to_proto_impl(self, context, **kwargs):
        model_type('recurrent_nn')
        RecurrentLayerGroupWithoutOutLinksBegin(
            name=self.__recurrent_name__,
            in_links=map(lambda x: x.name, self.__parents__))
        return self


class RecurrentLayerOutput(Layer):
    def __init__(self, recurrent_name, index, parent_layers):
        assert len(parent_layers) == 1
        self.__parents__ = parent_layers.values()[0]
        super(RecurrentLayerOutput, self).__init__(
            name=self.__parents__[index].name, parent_layers=parent_layers)
        self.__recurrent_name__ = recurrent_name

    def context_name(self):
        return self.__recurrent_name__ + ".end"

    def to_proto_impl(self, **kwargs):
        for l in self.__parents__:
            RecurrentLayerGroupSetOutLink(l.name)
        RecurrentLayerGroupEnd(name=self.__recurrent_name__)


LayerV2 = Layer
data = DataLayerV2
AggregateLevel = conf_helps.layers.AggregateLevel
ExpandLevel = conf_helps.layers.ExpandLevel
memory = MemoryV2


def __layer_name_mapping__(inname):
    if inname in ['data_layer', 'memory', 'mixed_layer', 'recurrent_group']:
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
        lambda x: x in ['input1', 'input2', 'label', 'input', 'a', 'b',
                        'expand_as',
                        'weights', 'vectors', 'weight', 'score', 'left',
                        'right', 'output_mem'],
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


@wrap_name_default()
def recurrent_group(step, input, name=None):
    if not isinstance(input, collections.Sequence):
        input = [input]

    non_static_inputs = filter(lambda x: not isinstance(x, StaticInputV2),
                               input)
    actual_input = [
        RecurrentLayerInput(
            recurrent_name=name,
            index=i,
            parent_layers={'recurrent_inputs': non_static_inputs})
        for i in xrange(len(non_static_inputs))
    ]

    def __real_step__(*args):
        rnn_input = list(args)
        static_inputs = filter(lambda x: isinstance(x, StaticInputV2), input)
        for static_input in static_inputs:
            mem_name = "__%s_memory__" % static_input.input.name
            mem = memory(
                name=mem_name,
                is_seq=static_input.is_seq,
                size=static_input.input.calculate_size,
                boot_layer=static_input.input)
            with mixed(
                    name=mem_name,
                    size=static_input.input.calculate_size,
                    act=activation.Identity()) as mix:
                mix += identity_projection(input=mem)
            rnn_input.insert(input.index(static_input), mix)
        return step(*rnn_input)

    actual_output = __real_step__(*actual_input)

    if not isinstance(actual_output, collections.Sequence):
        actual_output = [actual_output]

    retv = [
        RecurrentLayerOutput(
            recurrent_name=name,
            index=i,
            parent_layers={'recurrent_outputs': actual_output})
        for i in xrange(len(actual_output))
    ]
    if len(retv) == 1:
        return retv[0]
    else:
        return retv
