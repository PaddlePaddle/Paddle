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
`paddle.v2.layer` is a part of model config packages in paddle.v2. In API v2,
we want to make Paddle a plain Python package. The model config package defined
the way how to configure a neural network topology in Paddle Python code.

The primary usage shows below.

..  code-block:: python

    import paddle.v2 as paddle

    img = paddle.layer.data(name='img', type=paddle.data_type.dense_vector(784))
    hidden = paddle.layer.fc(input=img, size=200)
    prediction = paddle.layer.fc(input=hidden, size=10,
                                 act=paddle.activation.Softmax())

    # use prediction instance where needed.
    parameters = paddle.parameters.create(cost)
"""

import collections
import inspect
import re

import paddle.trainer_config_helpers as conf_helps
from paddle.trainer.config_parser import \
    RecurrentLayerGroupWithoutOutLinksBegin, RecurrentLayerGroupSetOutLink, \
    RecurrentLayerGroupEnd, model_type
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as __parse__
from paddle.trainer_config_helpers.default_decorators import wrap_act_default
from paddle.trainer_config_helpers.default_decorators import \
    wrap_bias_attr_default
from paddle.trainer_config_helpers.default_decorators import wrap_name_default
from paddle.trainer_config_helpers.layers import RecurrentLayerGroupSetGenerator, Generator
from paddle.trainer_config_helpers.layers import layer_support

import activation
import attr
import data_type
from config_base import Layer, __convert_to_v2__

__all__ = ['parse_network', 'data']


def parse_network(output_layers, extra_layers=None):
    """
    Parse all layers in the neural network graph and
    then generate a ModelConfig object.

    ..  note::

        This function is used internally in paddle.v2 module. User should never
        invoke this method.

    :param output_layers: Output layers.
    :type output_layers: Layer
    :param extra_layers: Some layers in the neural network graph are not in the
                         path of output_layers.
    :type extra_layers: Layer
    :return: A ModelConfig object instance.
    :rtype: ModelConfig
    """
    if not isinstance(output_layers, collections.Sequence):
        output_layers = [output_layers]
    if extra_layers is not None and not isinstance(extra_layers,
                                                   collections.Sequence):
        extra_layers = [extra_layers]

    def __real_func__():
        """
        __real_func__ is the function that config_parser.parse invoked. It is
        the plain old paddle configuration function.
        """
        context = dict()
        real_output = [each.to_proto(context=context) for each in output_layers]
        if extra_layers is not None:
            extra_output = [
                each.to_proto(context=context) for each in extra_layers
            ]
        conf_helps.outputs(real_output)

    return __parse__(__real_func__)


"""
Some layer may need some special config, and can not use __convert_to_v2__ to convert.
So we also need to implement some special LayerV2.
"""


class DataLayerV2(Layer):
    METHOD_NAME = 'data_layer'

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

    def __map_docstr__(doc):
        doc = re.sub(r'(data = [^\)]+)\).*',
                     "data = paddle.layer.data(name=\"input\", "
                     "type=paddle.data_type.dense_vector(1000))", doc)

        doc = re.sub(r':param size:.*',
                     ':param type: Data type of this data layer', doc)
        doc = re.sub(r':type size:.*',
                     ":type size: paddle.v2.data_type.InputType", doc)
        return doc


class MemoryV2(Layer):
    def __init__(self, name, extra_input=None, **kwargs):
        """
        Init memory object, if memory is inited inside recurrent_group step
        function, it may depend on a boot_layer that should be initialized
        outside recurrent_group, so we:
            1. add RecurrentLayerInput to extra_parent of self.
            2. add boot_layer to the extra_parent of RecurrentLayerInput.

        :param extra_input: list of RecurrentLayerInput
        :type extra_input: [RecurrentLayerInput]
        """
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
                extra.append_extra_parent(kwargs['boot_layer'])
                self.__boot_layer_name__ = kwargs['boot_layer'].name

    def to_proto_impl(self, **kwargs):
        args = dict()
        for each in kwargs:
            args[each] = kwargs[each]
        for each in self.__kwargs__:
            args[each] = self.__kwargs__[each]

        if self.__boot_layer_name__ is not None:
            args['boot_layer'] = self.__context__[self.__boot_layer_name__]

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


class StaticInputV2(object):
    def __init__(self, input, is_seq=False, size=None):
        assert isinstance(input, LayerV2)
        self.name = input.name
        self.input = input
        self.is_seq = is_seq
        self.size = size
        # TODO(add size check)
        # assert input.size is not None or size is not None


class BaseGeneratedInputV2(object):
    def __init__(self):
        self.bos_id = None
        self.eos_id = None

    def before_real_step(self):
        raise NotImplementedError()

    def after_real_step(self, *args):
        raise NotImplementedError()


class GeneratedInputV2(BaseGeneratedInputV2):
    def __init__(self, size, embedding_name, embedding_size):
        super(GeneratedInputV2, self).__init__()
        self.size = size
        self.embedding_name = embedding_name
        self.embedding_size = embedding_size

    def after_real_step(self, input):
        return max_id(input=input, name='__beam_search_predict__')

    def before_real_step(self):
        predict_id = memory(
            name='__beam_search_predict__',
            size=self.size,
            boot_with_const_id=self.bos_id)

        trg_emb = embedding(
            input=predict_id,
            size=self.embedding_size,
            param_attr=attr.ParamAttr(name=self.embedding_name))
        return trg_emb


class RecurrentLayerGroupSetGeneratorV2(Layer):
    def __init__(self, eos_name, max_length, beam_size, num_results_per_sample):
        self.eos_name = eos_name
        self.max_length = max_length
        self.beam_size = beam_size
        self.num_results_per_sample = num_results_per_sample
        super(RecurrentLayerGroupSetGeneratorV2, self).__init__(
            name=eos_name, parent_layers={})

    def to_proto_impl(self, **kwargs):
        RecurrentLayerGroupSetGenerator(
            Generator(
                eos_layer_name=self.eos_name,
                max_num_frames=self.max_length,
                beam_size=self.beam_size,
                num_results_per_sample=self.num_results_per_sample))
        return self

    def context_name(self):
        return self.eos_name + ".fake"

    def use_context_name(self):
        return True


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


mixed.__doc__ = conf_helps.mixed_layer.__doc__


class RecurrentLayerInput(Layer):
    def __init__(self, recurrent_name, index, parent_layers):
        parents_len = len(parent_layers)
        assert parents_len <= 1
        if parents_len == 0:
            self.__parents__ = []
        else:
            self.__parents__ = parent_layers.values()[0]
        self.__recurrent_name__ = recurrent_name
        name = self.__parents__[
            index].name if index >= 0 else self.context_name()
        super(RecurrentLayerInput, self).__init__(
            name=name, parent_layers=parent_layers)

    def context_name(self):
        return self.__recurrent_name__ + ".begin"

    def to_proto_impl(self, **kwargs):
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
data.__name__ = 'data'
AggregateLevel = conf_helps.layers.AggregateLevel
ExpandLevel = conf_helps.layers.ExpandLevel
memory = MemoryV2
memory.__name__ = 'memory'
memory.__doc__ = conf_helps.memory.__doc__


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
    globals()[new_name].__name__ = new_name


for each_layer_name in dir(conf_helps):
    new_name = __layer_name_mapping__(each_layer_name)
    if new_name is not None:
        parent_names = __layer_name_mapping_parent_names__(each_layer_name)
        assert len(parent_names) != 0, each_layer_name
        __convert_layer__(new_name, each_layer_name, parent_names)

del parent_names
del new_name
del each_layer_name


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

    extra_input = None
    if len(non_static_inputs) == 0:
        extra_input = RecurrentLayerInput(
            recurrent_name=name, index=-1, parent_layers={})

    def __real_step__(*args):
        rnn_input = list(args)
        static_inputs = filter(lambda x: isinstance(x, StaticInputV2), input)
        for static_input in static_inputs:
            mem_name = "__%s_memory__" % static_input.input.name
            mem = memory(
                name=mem_name,
                extra_input=extra_input,
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


recurrent_group.__doc__ = conf_helps.recurrent_group.__doc__


@wrap_name_default()
def beam_search(step,
                input,
                bos_id,
                eos_id,
                beam_size,
                max_length=500,
                name=None,
                num_results_per_sample=None):
    if num_results_per_sample is None:
        num_results_per_sample = beam_size
    assert num_results_per_sample <= beam_size
    # logger.warning("num_results_per_sample should be less than beam_size")

    if isinstance(input, StaticInputV2) or isinstance(input,
                                                      BaseGeneratedInputV2):
        input = [input]

    generated_input_index = -1

    real_input = []
    for i, each_input in enumerate(input):
        assert isinstance(each_input, StaticInputV2) or isinstance(
            each_input, BaseGeneratedInputV2)
        if isinstance(each_input, BaseGeneratedInputV2):
            assert generated_input_index == -1
            generated_input_index = i
        else:
            real_input.append(each_input)

    assert generated_input_index != -1

    gipt = input[generated_input_index]
    assert isinstance(gipt, BaseGeneratedInputV2)

    gipt.bos_id = bos_id
    gipt.eos_id = eos_id

    def __real_step__(*args):
        eos_name = "__%s_eos_layer__" % name
        generator = RecurrentLayerGroupSetGeneratorV2(
            eos_name, max_length, beam_size, num_results_per_sample)

        args = list(args)
        before_step_layer = gipt.before_real_step()
        before_step_layer.append_child(
            layer=generator, parent_names=[before_step_layer.name])
        args.insert(generated_input_index, before_step_layer)

        predict = gipt.after_real_step(step(*args))

        eos_layer = eos(input=predict, eos_id=eos_id, name=eos_name)
        predict.append_child(layer=eos_layer, parent_names=[predict.name])

        return predict

    # tmp = paddle.layer.recurrent_group(
    #     step=__real_step__,
    #     input=real_input,
    #     reverse=False,
    #     name=name,
    #     is_generating=True)
    tmp = recurrent_group(step=__real_step__, input=real_input, name=name)

    return tmp


beam_search.__doc__ = conf_helps.beam_search.__doc__

__projection_names__ = filter(lambda x: x.endswith('_projection'),
                              dir(conf_helps))

__all__ += __projection_names__

__operator_names__ = filter(lambda x: x.endswith('_operator'), dir(conf_helps))
__all__ += __operator_names__

# convert projection
for prj in __projection_names__:
    globals()[prj] = __convert_to_v2__(
        prj, parent_names=['input'], is_default_name=False)
    globals()[prj].__name__ = prj

# convert operator
operator_list = [
    # [V1_method_name, parent_names],
    ['dotmul_operator', ['a', 'b']],
    ['conv_operator', ['img', 'filter']]
]
for op in operator_list:
    globals()[op[0]] = __convert_to_v2__(
        op[0], parent_names=op[1], is_default_name=False)
    globals()[op[0]].__name__ = op[0]
