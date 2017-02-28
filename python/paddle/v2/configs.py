import paddle.trainer_config_helpers as conf_helps
import paddle.trainer.PyDataProvider2 as pydp2
import collections
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as __parse__
from paddle.trainer_config_helpers.default_decorators import wrap_act_default
from paddle.trainer_config_helpers.default_decorators import \
    wrap_bias_attr_default
from paddle.trainer_config_helpers.default_decorators import wrap_name_default
from paddle.trainer_config_helpers.layers import layer_support

__all__ = ['activation', 'attr', 'data_type', 'pooling']


class Namespace(object):
    # emulate namespace
    pass


activation = Namespace()
attr = Namespace()
data_type = Namespace()
pooling = Namespace()
layer = Namespace()


def __copy__(obj, origin_name, dest_name=None, origin_module=conf_helps):
    if dest_name is None:
        dest_name = origin_name
    origin_obj = getattr(origin_module, origin_name)
    setattr(obj, dest_name, origin_obj)


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


layer.parse_network = parse_network


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
        assert isinstance(type, pydp2.InputType)

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
        return getattr(conf_helps, self.__method_name__)(**args)


@wrap_name_default("mixed")
@wrap_act_default(act=conf_helps.LinearActivation())
@wrap_bias_attr_default(has_bias=False)
@layer_support(conf_helps.layers.ERROR_CLIPPING, conf_helps.layers.DROPOUT)
def mixed(size=0,
          name=None,
          input=None,
          act=None,
          bias_attr=False,
          layer_attr=None):
    return MixedLayerV2(size, input, name, act, bias_attr, layer_attr)


layer.LayerV2 = Layer
layer.data = DataLayerV2
layer.DataLayerV2 = DataLayerV2
layer.AggregateLevel = conf_helps.layers.AggregateLevel
layer.ExpandLevel = conf_helps.layers.ExpandLevel
layer.mixed = mixed


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
        lambda x: x in ['input1', 'input2', 'label', 'input', 'a', 'b',
                        'expand_as',
                        'weights', 'vectors', 'weight', 'score', 'left',
                        'right'],
        all_args)


def __convert_layer__(_new_name_, _old_name_, _parent_names_):
    setattr(layer, _new_name_, __convert_to_v2__(_old_name_, _parent_names_))


def __initialize__():
    suffix = 'Activation'
    for act_name in filter(lambda x: x.endswith(suffix), dir(conf_helps)):
        __copy__(activation, act_name, act_name[:-len(suffix)])

    for attr_name in [
            'ParameterAttribute', 'ExtraLayerAttribute', 'ParamAttr',
            'ExtraAttr'
    ]:
        __copy__(attr, attr_name)

    for name in dir(pydp2):
        if name[0] != '_' and '_' in name and 'slot' not in name and \
                        'args' not in name:
            __copy__(data_type, name, origin_module=pydp2)

    suffix = 'Pooling'
    for pooling_name in filter(lambda x: x.endswith(suffix), dir(conf_helps)):
        __copy__(
            pooling,
            origin_name=pooling_name,
            dest_name=pooling_name[:-len(suffix)])

    for each_layer_name in dir(conf_helps):
        new_name = __layer_name_mapping__(each_layer_name)
        if new_name is not None:
            parent_names = __layer_name_mapping_parent_names__(each_layer_name)
            assert len(parent_names) != 0, each_layer_name
            __convert_layer__(new_name, each_layer_name, parent_names)

    # convert projection
    for prj in filter(lambda x: x.endswith('_projection'), dir(conf_helps)):
        setattr(
            layer,
            prj,
            __convert_to_v2__(
                prj, parent_names=['input'], is_default_name=False))

    # convert operator
    operator_list = [
        # [V1_method_name, parent_names],
        ['dotmul_operator', ['a', 'b']],
        ['conv_operator', ['img', 'filter']]
    ]
    for op in operator_list:
        setattr(
            layer,
            op[0],
            __convert_to_v2__(
                op[0], parent_names=op[1], is_default_name=False))


__initialize__()
