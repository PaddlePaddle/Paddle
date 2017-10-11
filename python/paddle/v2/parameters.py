import numpy as np
import py_paddle.swig_paddle as api
from paddle.proto.ParameterConfig_pb2 import ParameterConfig
import struct
import tarfile
import cStringIO
from topology import Topology

__all__ = ['Parameters', 'create']


def create(layers):
    """
    Create parameter pool by topology.

    :param layers:
    :return:
    """
    topology = Topology(layers)
    pool = Parameters()
    for param in topology.proto().parameters:
        pool.__append_config__(param)
    return pool


class Parameters(object):
    """
    Parameters is a dictionary contains Paddle's parameter. The key of
    Parameters is the name of parameter. The value of Parameters is a plain
    :code:`numpy.ndarry` .

    Basically usage is

    ..  code-block:: python

        data = paddle.layers.data(...)
        ...
        out = paddle.layers.fc(...)

        parameters = paddle.parameters.create(out)

        parameter_names = parameters.names()
        fc_mat = parameters.get('fc')
        print fc_mat
    """

    def __init__(self):
        self.__param_conf__ = dict()
        self.__gradient_machines__ = []
        self.__tmp_params__ = []

    def __append_config__(self, param_conf):
        """
        Append a parameter configuration. It used to initialize Parameters and
        should be invoked only in paddle.parameters.create

        :param param_conf: The parameter configuration in protobuf
        :type param_conf: ParameterConfig
        :return: Nothing
        """

        if not isinstance(param_conf, ParameterConfig):
            raise ValueError("param_conf must be paddle.proto.ParameterConfig")

        if param_conf.name in self.__param_conf__:
            raise ValueError("duplicated parameter %s" % param_conf.name)

        self.__param_conf__[param_conf.name] = param_conf

    def keys(self):
        """
        keys are the names of each parameter.

        :return: list of parameter name
        :rtype: list
        """
        return self.__param_conf__.keys()

    def names(self):
        """
        names of each parameter.

        :return: list of parameter name
        :rtype: list
        """
        return self.keys()

    def has_key(self, key):
        """
        has_key return true if there are such parameter name == key

        :param key: Parameter name
        :type key: basestring
        :return: True if contains such key
        """
        return key in self.__param_conf__.keys()

    def __iter__(self):
        """
        Return an iterator of parameter name. It is used by `for loop`
        or `in` operator.

        ..  code-block:: python

            parameters = paddle.parameters.create(...)
            if "fc_param" in parameters:
                print 'OK'
        :return: an iterator of parameter name
        :rtype: iterator
        """
        return iter(self.__param_conf__)

    def __getitem__(self, key):
        """
        Get parameter by parameter name. It uses Python dict syntax.

        :note: It will always copy the parameter from C++ side.
        :param key: Parameter name
        :type key: basestring
        :return: parameter value
        :rtype: np.ndarray
        """
        shape = self.get_shape(key)

        if len(self.__gradient_machines__) == 0:
            # create new parameter in python numpy.
            if len(self.__tmp_params__) != 0:
                ret_list = [
                    mat for name, mat in self.__tmp_params__ if name == key
                ]
                if len(ret_list) == 1:
                    return ret_list[0]
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
        """
        get shape of the parameter.

        :param key: parameter name
        :type key: basestring
        :return: parameter's shape
        :rtype: tuple
        """
        if not isinstance(key, basestring):
            raise ValueError("parameter name should be string")
        if not self.has_key(key):
            raise ValueError("No such parameter %s" % key)
        conf = self.__param_conf__[key]
        dims = conf.dims if conf.dims else (1, conf.size)
        return tuple(map(int, dims))

    def __setitem__(self, key, value):
        """
        Set parameter by parameter name & value. It use Python dict syntax.

        :note: It will always copy the parameter to C++ side.
        :param key: Parameter name
        :type key: basestring
        :param value: Parameter matrix.
        :type value: np.ndarray
        :return: Nothing
        """

        if not isinstance(value, np.ndarray):
            raise ValueError("Must return ndarray")
        value = value.astype(dtype=np.float32)
        shape = self.get_shape(key)
        if value.shape != shape:
            raise ValueError("Value shape mismatch, expect %s, should %s" %
                             (shape, value.shape))

        if len(self.__gradient_machines__) == 0:
            self.__tmp_params__.append((key, value))
        else:
            for each_gradient_machine in self.__gradient_machines__:
                __copy_parameter_to_gradient_machine__(each_gradient_machine,
                                                       key, value)

    def get(self, parameter_name):
        """
        Get parameter by parameter name.

        :note: It will always copy the parameter from C++ side.
        :param parameter_name: parameter name
        :type parameter_name: basestring
        :return: The parameter matrix.
        :rtype: np.ndarray
        """
        return self.__getitem__(key=parameter_name)

    def set(self, parameter_name, value):
        """
        Set parameter by parameter name & matrix.

        :param parameter_name: parameter name
        :type parameter_name: basestring
        :param value: parameter matrix
        :type value: np.ndarray
        :return: Nothing.
        """
        self.__setitem__(key=parameter_name, value=value)

    def append_gradient_machine(self, gradient_machine):
        """
        append gradient machine to parameters. This method is used internally in
        Trainer.train.

        :param gradient_machine: Paddle C++ GradientMachine object.
        :type gradient_machine: api.GradientMachine
        :return:
        """

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

    def serialize(self, name, f):
        """

        :param name:
        :param f:
        :type f: file
        :return:
        """
        param = self.get(name)
        size = reduce(lambda a, b: a * b, param.shape)
        f.write(struct.pack("IIQ", 0, 4, size))
        param = param.astype(np.float32)
        f.write(param.tobytes())

    def deserialize(self, name, f):
        """

        :param name:
        :param f:
        :type f: file
        :return:
        """
        f.read(16)  # header
        arr = np.frombuffer(f.read(), dtype=np.float32)
        self.set(name, arr.reshape(self.get_shape(name)))

    def to_tar(self, f):
        tar = tarfile.TarFile(fileobj=f, mode='w')
        for nm in self.names():
            buf = cStringIO.StringIO()
            self.serialize(nm, buf)
            tarinfo = tarfile.TarInfo(name=nm)
            buf.seek(0)
            tarinfo.size = len(buf.getvalue())
            tar.addfile(tarinfo, buf)

            conf = self.__param_conf__[nm]
            confStr = conf.SerializeToString()
            tarinfo = tarfile.TarInfo(name="%s.protobuf" % nm)
            tarinfo.size = len(confStr)
            buf = cStringIO.StringIO(confStr)
            buf.seek(0)
            tar.addfile(tarinfo, fileobj=buf)

    @staticmethod
    def from_tar(f):
        params = Parameters()
        tar = tarfile.TarFile(fileobj=f, mode='r')
        for finfo in tar:
            assert isinstance(finfo, tarfile.TarInfo)
            if finfo.name.endswith('.protobuf'):
                f = tar.extractfile(finfo)
                conf = ParameterConfig()
                conf.ParseFromString(f.read())
                params.__append_config__(conf)

        for param_name in params.names():
            f = tar.extractfile(param_name)
            params.deserialize(param_name, f)
        return params


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
