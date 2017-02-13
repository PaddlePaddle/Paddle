import numpy as np

from paddle.proto.ModelConfig_pb2 import ModelConfig
from paddle.proto.ParameterConfig_pb2 import ParameterConfig

__all__ = ['IParameterPool', 'create', 'ParameterFlag']


class ParameterFlag(object):
    """
    The flag for IParameterPool.get_parameter. If writeable, operation on return
    numpy array will also apply to Paddle parameter. But it will be slower in
    GPU mode.
    """
    READ_ONLY = 0x01
    WRITE_ONLY = 0x02
    READ_WRITE = READ_ONLY | WRITE_ONLY


class IParameterPool(object):
    """
    Interface of Parameter Pool. The parameter pool is a dictionary of
    parameters. User can modify parameter or customize parameter value
    by `get_parameter`.

    ..  code-block:: python

        pool = paddle.parameters.create(topo1, topo2)

        embedding = pool.get_parameter("embedding")
        assert isinstance(embedding, numpy.ndarray)
        print embedding[1:]
    """

    def get_parameter(self, name, flag=ParameterFlag.READ_WRITE):
        """
        Get a parameter by name.

        :param name: parameter name.
        :type name: basestring
        :param flag: the flag for return value. readable or writable.
        :type flag: int
        :return: The parameter value
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    def get_names(self):
        """
        Get all parameter names
        :return: all parameter names
        :rtype: list
        """
        raise NotImplementedError()


class NumpyParameterPool(IParameterPool):
    def __init__(self):
        self.__param_configs__ = dict()
        self.__params__ = dict()

    def append(self, conf):
        if not isinstance(conf, ParameterConfig):
            raise ValueError("conf must be ParameterConfig")

        if not conf.IsInitialized():
            raise ValueError("conf is not initialized")

        self.__param_configs__[conf.name] = conf
        self.__params__[conf.name] = None

    def get_config(self, name):
        if name not in self.__param_configs__:
            raise ValueError("parameter %s is not appended" % name)

        return self.__param_configs__[name]

    def get_parameter(self, name, *args, **kwargs):
        if name not in self.__params__:
            raise ValueError("parameter %s is not appended" % name)

        param = self.__params__[name]
        if param is None:
            shape = self.__param_configs__[name].dims
            if len(shape) == 0:
                raise ValueError("parameter %s is no shape" % name)
            param = np.ndarray(
                shape=[int(item) for item in shape], dtype='float32')
            self.__params__[name] = param
        return param

    def get_names(self):
        return self.__param_configs__.keys()


def create(*topologies):
    """
    Create parameter pool by topologies.

    :param topologies:
    :return:
    """
    pool = NumpyParameterPool()
    for topo in topologies:
        if not isinstance(topo, ModelConfig):
            raise ValueError(
                'create must pass a topologies which type is ModelConfig')

        for param in topo.parameters:
            pool.append(param)
    return pool
