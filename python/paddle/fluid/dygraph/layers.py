# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import contextlib
import sys
import numpy as np
import collections
import six
from .. import unique_name
from paddle.fluid import core
from .layer_object_helper import LayerObjectHelper
from paddle.fluid import framework
from ..param_attr import ParamAttr

__all__ = ['Layer', 'PyLayer']


class Layer(core.Layer):
    """Layers composed of operators.

    Args:
        name_scope: prefix name used by the layer to name parameters.
            If prefix is "my_model/layer_1", parameter name in MyLayer
            can be "my_model/layer_1/MyLayer/w_n", where w is the parameter
            base name and n is an unique suffix auto-generated.
        dtype: data type for the variables in the layer.
    """

    def __init__(self, name_scope, dtype=core.VarDesc.VarType.FP32):
        self._full_name = unique_name.generate(name_scope + "/" +
                                               self.__class__.__name__)
        self._built = False
        self._dtype = dtype
        self._parameters = collections.OrderedDict()
        self._sub_layers = collections.OrderedDict()

        self._helper = LayerObjectHelper(self._full_name)

    def full_name(self):
        """Full name for this layers.

          Full name is composed by name_scope + "/" + MyLayer.__class__.__name__

        Returns full name of this name.
        """
        return self._full_name

    def create_parameter(self,
                         attr,
                         shape,
                         dtype,
                         is_bias=False,
                         default_initializer=None):
        """Create parameters for this layers.

           Args:
               attr: [ParamAttr] should be the parameter attribute for this parameter
               shape: shape of the paramter
               dtype: data type of this parameter
               is_bias: if this is a bias parameter
               default_initializer: set the default initializer for this parameter

        Returns created parameter Variable.
        """
        if isinstance(attr, ParamAttr) and (attr.name is not None):
            attr.name = ".".join([self._full_name, attr.name])
        elif isinstance(attr, six.string_types):
            attr = ".".join([self._full_name, attr])
        return self._helper.create_parameter(attr, shape, dtype, is_bias,
                                             default_initializer)

    # TODO: Add more parameter list when we need them
    def create_variable(self,
                        name=None,
                        persistable=None,
                        dtype=None,
                        type=core.VarDesc.VarType.LOD_TENSOR):
        """Create Variable for this layers.

           Args:
               name: name of the variable
               persistable: if set this variable persistable
               dtype: data type of data in the variable
               type: type of the variable

        Returns created Variable.
        """
        if name is not None:
            var_name = ".".join([self._full_name, name])
        else:
            var_name = unique_name.generate(".".join(
                [self._full_name, "_generated_var"]))

        return self._helper.main_program.current_block().create_var(
            name=var_name, persistable=persistable, dtype=dtype, type=type)

    def parameters(self, include_sublayers=True):
        """Returns a list of Parameters from current and sub-layers.

        Args:
            include_sublayers: If true, also include the parameters from
            sublayers.

        Returns a list of Parameters.
        """
        ret = [p for p in self._parameters.values()]
        if include_sublayers:
            for l in self._sub_layers.values():
                for p in l.parameters(include_sublayers):
                    ret.append(p)
        return ret

    def sublayers(self, include_sublayers=True):
        """Returns a list of sub layers.

        Args:
            include_sublayers: If true, also include the layers from sublayers.

        Returns a list of sub layers.
        """
        ret = [l for l in self._sub_layers.values()]
        if include_sublayers:
            for l in self._sub_layers.values():
                for sub_l in l.sublayers(include_sublayers):
                    ret.append(sub_l)
        return ret

    def clear_gradients(self):
        for p in self.parameters():
            p.clear_gradient()

    def _build_once(self, *args):
        pass

    def __call__(self, *inputs):
        if not self._built:
            self._build_once(*inputs)

        outputs = self.forward(*inputs)
        self._built = True
        return outputs

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *inputs):
        raise ValueError("Layer shouldn't implement backward")

    def add_sublayer(self, name, sublayer):
        """Adds a sub Layer instance.

          Added sublayer can be access like self.name.

        Args:
            name: name of this sublayer.
            sublayer: an instance of Layer.
        Returns:
            the sublayer passed in.
        """
        assert isinstance(sublayer, core.Layer)

        self._sub_layers[name] = sublayer
        return sublayer

    def add_parameter(self, name, parameter):
        """Adds a Parameter instance.

          Added parameter can be access like self.name.

        Args:
            name: name of this sublayer.
            parameter: an instance of Parameter.
        Returns:
            the parameter passed in.
        """
        assert isinstance(parameter, framework.Parameter)
        self._parameters[name] = parameter
        return parameter

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._sub_layers:
            return self._sub_layers[name]

    def __setattr__(self, name, value):
        if isinstance(value, framework.Parameter):
            params = self.__dict__.get('_parameters', None)
            if params is None:
                raise ValueError(
                    "super(YourLayer, self).__init__() should be called first")
            params[name] = value
        elif isinstance(value, core.Layer):
            layers = self.__dict__.get('_sub_layers', None)
            if layers is None:
                raise ValueError(
                    "super(YourLayer, self).__init__() should be called first")
            layers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._sub_layers:
            del self._sub_layers[name]
        else:
            object.__delattr__(self, name)

    def state_dict(self, destination=None, prefix='', include_sublayers=True):
        if destination is None:
            destination = collections.OrderedDict()
        for name, data in self._parameters.items():
            if data is not None:
                destination[prefix + name] = data

        if include_sublayers:
            for layer_name, layer_item in self._sub_layers.items():
                if layer_item is not None:
                    destination_temp = destination.copy()
                    destination_temp.update(
                        layer_item.state_dict(destination_temp, prefix +
                                              layer_name + ".",
                                              include_sublayers))
                    destination = destination_temp
        return destination

    def load_dict(self, stat_dict, include_sublayers=True):
        for name, item in self.__dict__.get('_parameters', None).items():
            if item.name in stat_dict:
                self.__setattr__(name, stat_dict[item.name])

        if include_sublayers:
            for layer_name, layer_item in self._sub_layers.items():
                if layer_item is not None:
                    layer_item.load_dict(stat_dict)


class PyLayer(core.PyLayer):
    """Layers composed of user-defined python codes."""

    def __init__(self):
        super(PyLayer, self).__init__()

    @classmethod
    def _do_forward(cls, inputs):
        return cls._to_tuple(cls.forward(inputs))

    @classmethod
    def _do_backward(cls, inputs):
        return cls._to_tuple(cls.backward(inputs))

    @staticmethod
    def _to_tuple(inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, tuple):
            inputs = [inputs]
        ret = []
        for inp in inputs:
            tensor = core.LoDTensor()
            tensor.set(inp, core.CPUPlace())
            ret.append(tensor)
        return tuple(ret)

    @staticmethod
    def forward(*inputs):
        raise NotImplementedError

    @staticmethod
    def backward(*douts):
        raise NotImplementedError

    @classmethod
    def __call__(cls, *inputs):
        tracer = framework._dygraph_tracer()
        block = framework.default_main_program().current_block()
        ivar_inputs = [x._ivar for x in inputs]

        if not hasattr(cls, 'forward_id'):
            cls.forward_id = core.PyLayer.num_funcs() + 1
            PyLayer.register_func(cls.forward_id, cls._do_forward)
            cls.backward_id = core.PyLayer.num_funcs() + 1
            PyLayer.register_func(cls.backward_id, cls._do_backward)

        iop = core.OpBase(cls.__class__.__name__ + str(cls.forward_id))
        iop.forward_id = cls.forward_id
        iop.backward_id = cls.backward_id
        block.ops.append(iop)
        ivars = tracer.py_trace(iop, ivar_inputs, False)
        ret = []
        for ivar in ivars:
            tensor = ivar.value().get_tensor()
            py_var = framework.Variable(
                block,
                type=core.VarDesc.VarType.LOD_TENSOR,
                name=None,
                shape=tensor.shape(),
                dtype=tensor._dtype(),
                ivar=ivar)
            ret.append(py_var)
        return ret
