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

import numpy as np
from paddle.proto.ParameterConfig_pb2 import ParameterConfig
from collections import OrderedDict
import paddle.trainer.config_parser as cp
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
    initializers = cp.g_parameter_initializer_map
    for param in topology.proto().parameters:
        pool.__append_config__(param)
        if param.name in initializers:
            pool[param.name] = initializers[param.name](param.name)
    return pool


class Parameters(object):
    """
    `Parameters` manages all the learnable parameters in a neural network.
    It stores parameters' information in an OrderedDict. The key is
    the name of a parameter, and value is a parameter's configuration(in
    protobuf format), such as initialization mean and std, its size, whether it
    is a static parameter, and so on.

    :param __param_conf__: store the configurations of learnable parameters in
        the network in an OrderedDict. Parameter is added one by one into the
        dict by following their created order in the network: parameters of
        the previous layers in a network are careted first. You can visit the
        parameters from bottom to top by iterating over this dict.
    :type __param_conf__: OrderedDict
    :param __gradient_machines__: all of the parameters in a neural network are
        appended to a PaddlePaddle gradient machine, which is used internally to
        copy parameter values between C++ and Python end.
    :type __gradient_machines__: list
    :param __tmp_params__: a dict to store dummy parameters if no
        __gradient_machines__ is appended to `Parameters`.
    :type __tmp_params__: dict

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
        self.__param_conf__ = OrderedDict()
        self.__gradient_machines__ = []
        self.__tmp_params__ = dict()

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

    def update_param_conf(self, model_config):
        for p in model_config.parameters:
            self.__param_conf__[p.name] = p

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

    def __getter_inner(self, key, param_type):
        import py_paddle.swig_paddle as api
        shape = self.get_shape(key)

        if len(self.__gradient_machines__) == 0:
            # create new parameter in python numpy.
            if key in self.__tmp_params__:
                return self.__tmp_params__[key]
            else:
                return np.ndarray(shape=shape, dtype=np.float32)
        else:
            for each_gradient_machine in self.__gradient_machines__:
                param = __get_parameter_in_gradient_machine__(
                    each_gradient_machine, key)
                # for simplify implementation now, we always copy from C++
                assert isinstance(param, api.Parameter)
                val = param.getBuf(param_type)
                assert isinstance(val, api.Vector)
                val = val.copyToNumpyArray()
                return val
                # else continue

            raise RuntimeError("Unexpected branch")

    def __getitem__(self, key):
        """
        Get parameter by parameter name. It uses Python dict syntax.

        :note: It will always copy the parameter from C++ side.
        :param key: Parameter name
        :type key: basestring
        :return: parameter value
        :rtype: np.ndarray
        """
        import py_paddle.swig_paddle as api
        return self.__getter_inner(key, api.PARAMETER_VALUE)

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
            self.__tmp_params__[key] = value
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

    def get_grad(self, key):
        """
        Get grandient by parameter name.

        :note: It will always copy the parameter from C++ side.
        :param key: parameter name
        :type key: basestring
        :return: The grandient matrix.
        :rtype: np.ndarray
        """
        import py_paddle.swig_paddle as api
        if self.__param_conf__[key].is_static:
            return np.zeros(self.__param_conf__[key].size, dtype=np.float32)

        return self.__getter_inner(key, api.PARAMETER_GRADIENT)

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

        :param gradient_machine: PaddlePaddle C++ GradientMachine object.
        :type gradient_machine: api.GradientMachine
        :return:
        """
        import py_paddle.swig_paddle as api
        if not isinstance(gradient_machine, api.GradientMachine):
            raise ValueError("gradient_machine should be api.GradientMachine")

        if len(self.__tmp_params__) != 0:
            for name, val in self.__tmp_params__.iteritems():
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
        s = param.tostring()
        wrote_size = 0
        buf = buffer(s, wrote_size, 65535)
        while buf:  # f.write crashes with big data blog.
            f.write(buf)
            wrote_size += 65535
            buf = buffer(s, wrote_size, 65535)

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
        """
        Save parameters to a tar file.

        WARNING: You should use `paddle.v2.trainer.SGD.save_parameter_to_tar(f)`
            to save parameters most of the time. Otherwise, some settings such
            as model average will not take effect.

        :param f:
        :type f: file
        :return:
        """
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
        """
        Create a `Parameters` object from the given file. And
        the `Parameters` only contains the parameters in this
        file. It is adapted the parameters are same in the
        defined network and the given file. For example, it
        can be used in the inference.

        :param f: the initialized model file.
        :type f: tar file
        :return: A Parameters object.
        :rtype: Parameters.
        """
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

    def init_from_tar(self, f, exclude_params=[]):
        """
        Different from `from_tar`, this interface can be used to
        init partial network parameters from another saved model.

        :param f: the initialized model file.
        :type f: tar file
        :param exclude_params: the names of parameters that should  
            not be initialized from the model file.
        :type exclude_params: list of strings
        :return: Nothing.
        """

        tar_param = Parameters.from_tar(f)
        for pname in tar_param.names():
            if pname in self.names() and pname not in exclude_params:
                self.set(pname, tar_param.get(pname))


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
    import py_paddle.swig_paddle as api
    param = __get_parameter_in_gradient_machine__(gradient_machine, name)
    vec = param.getBuf(api.PARAMETER_VALUE)
    assert isinstance(vec, api.Vector)
    vec.copyFromNumpyArray(arr.flatten())
