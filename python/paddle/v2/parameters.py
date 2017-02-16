import numpy as np

from paddle.proto.ModelConfig_pb2 import ModelConfig
from paddle.proto.ParameterConfig_pb2 import ParameterConfig
import py_paddle.swig_paddle as api

__all__ = ['Parameters', 'create']


def create(*topologies):
    """
    Create parameter pool by topologies.

    :param topologies:
    :return:
    """
    pool = Parameters()
    for topo in topologies:
        if not isinstance(topo, ModelConfig):
            raise ValueError(
                'create must pass a topologies which type is ModelConfig')

        for param in topo.parameters:
            pool.append_config(param)
    return pool


class Parameters(object):
    """
    The parameters
    """

    def __init__(self):
        self.__param_conf__ = dict()
        self.__gradient_machines__ = []
        self.__tmp_params__ = []

    def append_config(self, param_conf):
        if not isinstance(param_conf, ParameterConfig):
            raise ValueError("param_conf must be paddle.proto.ParameterConfig")

        if param_conf.name in self.__param_conf__:
            raise ValueError("duplicated parameter %s" % param_conf.name)

        self.__param_conf__[param_conf.name] = param_conf

    def keys(self):
        return self.__param_conf__.keys()

    def names(self):
        return self.keys()

    def has_key(self, key):
        return key in self.__param_conf__.keys()

    def __iter__(self):
        return iter(self.__param_conf__)

    def __getitem__(self, key):
        shape = self.get_shape(key)

        if len(self.__gradient_machines__) == 0:
            # create new parameter in python numpy.
            return np.ndarray(shape=shape, dtype=np.float32)
        else:
            for each_gradient_machine in self.__gradient_machines__:
                param = __get_parameter_in_gradient_machine__(
                    each_gradient_machine, key)
                # for simplify implementation now, we always copy from C++
                assert isinstance(param, api.Parameter)
                val = param.getBuf(api.PARAMETER_VALUE)
                assert isinstance(val, api.Vector)
                val = val.copyToNumpyArray()
                return val
                # else continue

            raise RuntimeError("Unexpected branch")

    def get_shape(self, key):
        if not isinstance(key, basestring):
            raise ValueError("parameter name should be string")
        if not self.has_key(key):
            raise ValueError("No such parameter %s" % key)
        conf = self.__param_conf__[key]
        return map(int, conf.dims)

    def __setitem__(self, key, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Must return ndarray")
        value = value.astype(dtype=np.float32)
        shape = self.get_shape(key)
        if not reduce(lambda a, b: a and b,
                      map(lambda x: x[0] == x[1], zip(value.shape, shape))):
            raise ValueError("Value shape mismatch, expect %s, should %s" %
                             (shape, value.shape))

        if len(self.__gradient_machines__) == 0:
            self.__tmp_params__.append((key, value))
        else:
            for each_gradient_machine in self.__gradient_machines__:
                __copy_parameter_to_gradient_machine__(each_gradient_machine,
                                                       key, value)

    def get(self, parameter_name):
        return self.__getitem__(key=parameter_name)

    def set(self, parameter_name, value):
        self.__setitem__(key=parameter_name, value=value)

    def append_gradient_machine(self, gradient_machine):
        if not isinstance(gradient_machine, api.GradientMachine):
            raise ValueError("gradient_machine should be api.GradientMachine")

        if len(self.__tmp_params__) != 0:
            for name, val in self.__tmp_params__:
                try:
                    __copy_parameter_to_gradient_machine__(gradient_machine,
                                                           name, val)
                except ValueError:
                    # If no such parameter in gradient machine, then don't copy
                    pass
            self.__gradient_machines__.append(gradient_machine)


def __get_parameter_in_gradient_machine__(gradient_machine, name):
    """

    :param gradient_machine:
    :type gradient_machine: api.GradientMachine
    :param name:
    :return:
    :rtype: api.Parameter
    """
    params = filter(lambda p: p.getName() == name,
                    gradient_machine.getParameters())

    if len(params) == 0:
        raise ValueError("No such parameter")
    elif len(params) > 1:
        raise ValueError("Unexpected branch")
    else:
        return params[0]


def __copy_parameter_to_gradient_machine__(gradient_machine, name, arr):
    """
    Copy a python ndarray into the gradient machine.

    :param gradient_machine:
    :type gradient_machine: api.GradientMachine
    :param name:
    :param arr:
    :type arr: np.ndarray
    :return:
    :rtype: api.Parameter
    """
    param = __get_parameter_in_gradient_machine__(gradient_machine, name)
    vec = param.getBuf(api.PARAMETER_VALUE)
    assert isinstance(vec, api.Vector)
    vec.copyFromNumpyArray(arr.flatten())
