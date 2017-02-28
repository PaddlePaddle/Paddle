import paddle.trainer_config_helpers as conf_helps
import paddle.trainer.PyDataProvider2 as pydp2

__all__ = ['activation', 'attr', 'data_type', 'pooling']


class Namespace(object):
    # emulate namespace
    pass


activation = Namespace()
attr = Namespace()
data_type = Namespace()
pooling = Namespace()


def __copy__(obj, origin_name, dest_name=None, origin_module=conf_helps):
    if dest_name is None:
        dest_name = origin_name
    origin_obj = getattr(origin_module, origin_name)
    setattr(obj, dest_name, origin_obj)


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
        if name[0] != '_' and '_' in name and 'slot' not in name and 'args' not in name:
            __copy__(data_type, name, origin_module=pydp2)

    suffix = 'Pooling'
    for pooling_name in filter(lambda x: x.endswith(suffix), dir(conf_helps)):
        __copy__(
            pooling,
            origin_name=pooling_name,
            dest_name=pooling_name[:-len(suffix)])


__initialize__()
