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
import re
from . import parallel_helper
from .. import unique_name
from paddle.fluid import core
from .layer_object_helper import LayerObjectHelper
from .base import program_desc_tracing_guard
from paddle.fluid import framework
from ..param_attr import ParamAttr
import copy
import warnings

__all__ = ['Layer']

_first_cap_re = re.compile('(.)([A-Z][a-z]+)')
_all_cap_re = re.compile('([a-z])([A-Z])')


def _convert_camel_to_snake(name):
    s1 = _first_cap_re.sub(r'\1_\2', name)
    return _all_cap_re.sub(r'\1_\2', s1).lower()


class Layer(core.Layer):
    """Dynamic graph Layer based on OOD, includes the parameters of the layer, the structure of the forward graph and so on.

    Parameters:
        name_scope (str, optional): prefix name used by the layer to name parameters.
            If prefix is "my_layer", parameter name in MyLayer
            can be "my_layer_0.w_n", where "w" is the parameter
            base name and "n" is an unique suffix auto-generated.
            If None, prefix name will be snake cased class name. Default: None.
        dtype(str or core.VarDesc.VarType, optional): data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16".
                Default: ``core.VarDesc.VarType.FP32``
    
    Returns:
        None
    """

    def __init__(self, name_scope=None, dtype=core.VarDesc.VarType.FP32):
        if name_scope is None:
            name_scope = _convert_camel_to_snake(self.__class__.__name__)
        self._full_name = unique_name.generate(name_scope)
        self._helper = LayerObjectHelper(self._full_name)
        self._built = False
        self._dtype = dtype

        self._parameters = collections.OrderedDict()
        self._sub_layers = collections.OrderedDict()
        self._loaddict_holder = collections.OrderedDict()

    def train(self):
        framework._dygraph_tracer().train_mode()

    def eval(self):
        framework._dygraph_tracer().eval_mode()

    def full_name(self):
        """Full name for this layer, composed by name_scope + "/" + MyLayer.__class__.__name__

        Returns:
            str: full name of this layer.
        """
        return self._full_name

    def create_parameter(self,
                         shape,
                         attr=None,
                         dtype='float32',
                         is_bias=False,
                         default_initializer=None):
        """Create parameters for this layer.
        
        Parameters:
            shape(list): Shape of the parameter.
            attr(ParamAttr, optional): Parameter attribute of weight. Please refer to :ref:`api_fluid_ParamAttr`. Default: None.
            dtype(str or core.VarDesc.VarType or str, optional): Data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16". Default: "float32".
            is_bias(bool, optional): if this is a bias parameter. Default: False.
            default_initializer(Initializer, optional): the default initializer for this parameter.
                If set None, default initializer will be set to :ref:`api_fluid_initializer_XavierInitializer` and :ref:`api_fluid_initializer_ConstantInitializer`
                for non-bias and bias parameter, respectively. Default: None.

        Returns:
            :ref:`api_guide_Variable_en` : created parameter.
        """
        temp_attr = copy.deepcopy(attr)
        if isinstance(temp_attr, six.string_types) and temp_attr == "":
            temp_attr = None
        return self._helper.create_parameter(temp_attr, shape, dtype, is_bias,
                                             default_initializer)

    # TODO: Add more parameter list when we need them
    def create_variable(self,
                        name=None,
                        persistable=None,
                        dtype=None,
                        type=core.VarDesc.VarType.LOD_TENSOR):
        """Create Variable for this layer.

        Parameters:
            name(str, optional): name of the variable. Please refer to :ref:`api_guide_Name` . Default: None
            persistable(bool, optional): if set this variable persistable. Default: False
            dtype(str or core.VarDesc.VarType, optional): data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16".
                If set None, it will be ``core.VarDesc.VarType.FP32``. Default: None
            type(core.VarDesc.VarType, optional): type of the variable. No need to set this parameter. Default: ``core.VarDesc.VarType.LOD_TENSOR``

        Returns:
            :ref:`api_guide_Variable_en` : created Variable.
        """
        if name is not None:
            var_name = ".".join([self._full_name, name])
        else:
            var_name = unique_name.generate(".".join(
                [self._full_name, "_generated_var"]))

        return self._helper.main_program.current_block().create_var(
            name=var_name, persistable=persistable, dtype=dtype, type=type)

    def parameters(self, include_sublayers=True):
        """Returns a list of all Parameters from current layer and its sub-layers.

        Parameters:
            include_sublayers(bool, optional): Whether include the parameters of sublayers. If True, also include the parameters from sublayers. Default: True

        Returns:
            list of :ref:`api_guide_Variable_en` : a list of Parameters.
        """
        ret = [
            param
            for _, param in self.named_parameters(
                include_sublayers=include_sublayers)
        ]
        return ret

    def sublayers(self, include_sublayers=True):
        """Returns a list of sub layers.

        Parameters:
            include_sublayers(bool, optional): Whether return the sublayers of sublayers. If True, also include the sublayers of sublayers. Default: True

        Returns:
            list of Layer : a list of sub layers.
        """
        ret = [
            layer
            for _, layer in self.named_sublayers(
                include_sublayers=include_sublayers)
        ]
        return ret

    def named_parameters(self, prefix='', include_sublayers=True):
        """
        Returns an iterator over all parameters in the Layer, yielding tuple of name and parameter.

        Parameters:
            prefix(str, optional): Prefix to prepend to all parameter names. Default: ''.
            include_sublayers(bool, optional): Whether include the parameters of sublayers.
                If True, also include the named parameters from sublayers. Default: True.

        Yields:
            (string, Parameter): Tuple of name and Parameter

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                with fluid.dygraph.guard():
                    fc1 = fluid.Linear(10, 3)
                    fc2 = fluid.Linear(3, 10, bias_attr=False)
                    model = fluid.dygraph.Sequential(fc1, fc2)
                    for name, param in model.named_parameters():
                        print(name, param)

        """
        params_set = set()
        named_sublayers = self.named_sublayers(
            prefix=prefix,
            include_sublayers=include_sublayers,
            include_self=True)
        for layer_prefix, sublayer in named_sublayers:
            params = sublayer._parameters.items()
            for key, param in params:
                if param is None or param in params_set:
                    continue
                params_set.add(param)
                name = layer_prefix + ('.' if layer_prefix else '') + key
                yield name, param

    def named_sublayers(self,
                        prefix='',
                        include_sublayers=True,
                        include_self=False,
                        layers_set=None):
        """
        Returns an iterator over all sublayers in the Layer, yielding tuple of name and sublayer.
        The duplicate sublayer will only be yielded once.

        Parameters:
            prefix(str, optional): Prefix to prepend to all parameter names. Default: ''.
            include_sublayers(bool, optional): Whether include the sublayers. Default: True.
            include_self(bool, optional): Whether include the Layer itself. Default: False.
            layers_set(set, optioanl): The set to record duplicate sublayers. Default: None.

        Yields:
            (string, Layer): Tuple of name and Layer

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                with fluid.dygraph.guard():
                    fc1 = fluid.Linear(10, 3)
                    fc2 = fluid.Linear(3, 10, bias_attr=False)
                    model = fluid.dygraph.Sequential(fc1, fc2)
                    for prefix, layer in model.named_sublayers():
                        print(prefix, layer)

        """
        if layers_set is None:
            layers_set = set()
        if include_self and self not in layers_set:
            layers_set.add(self)
            yield prefix, self
        if include_sublayers:
            for key, layer in self._sub_layers.items():
                if layer is None:
                    continue
                layer_prefix = prefix + ('.' if prefix else '') + key
                for p, l in layer.named_sublayers(
                        prefix=layer_prefix,
                        include_sublayers=include_sublayers,
                        include_self=True,
                        layers_set=layers_set):
                    yield p, l

    def clear_gradients(self):
        """
        Clear the gradients of all parameters for this layer.
        
        Returns:
            None
        
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                with fluid.dygraph.guard():
                    value = np.arange(26).reshape(2, 13).astype("float32")
                    a = fluid.dygraph.to_variable(value)
                    linear = fluid.Linear(13, 5, dtype="float32")
                    adam = fluid.optimizer.Adam(learning_rate=0.01, 
                                                parameter_list=linear.parameters())
                    out = linear(a)
                    out.backward()
                    adam.minimize(out)
                    linear.clear_gradients()

        """
        for p in self.parameters():
            if p.trainable:
                p.clear_gradient()

    def _build_once(self, *args, **kwargs):
        pass

    def __call__(self, *inputs, **kwargs):
        if not self._built:
            with program_desc_tracing_guard(False):
                self._build_once(*inputs, **kwargs)
                if parallel_helper._is_data_parallel_mode():
                    parallel_helper._broadcast_parameters(
                        self._parameters.values())
            self._built = True

        outputs = self.forward(*inputs, **kwargs)
        return outputs

    def forward(self, *inputs, **kwargs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.

        Parameters:
            *inputs(tuple): unpacked tuple arguments
            **kwargs(dict): unpacked dict arguments
        """
        raise NotImplementedError

    def backward(self, *inputs):
        raise ValueError("Layer shouldn't implement backward")

    def add_sublayer(self, name, sublayer):
        """Adds a sub Layer instance.

        Added sublayer can be accessed by self.name

        Parameters:
            name(str): name of this sublayer.
            sublayer(Layer): an instance of Layer.
        Returns:
            Layer: the sublayer passed in.
        """
        assert isinstance(sublayer, core.Layer)

        self._sub_layers[name] = sublayer
        return sublayer

    def add_parameter(self, name, parameter):
        """Adds a Parameter instance.

        Added parameter can be accessed by self.name

        Parameters:
            name(str): name of this sublayer.
            parameter(Parameter): an instance of Parameter.
        Returns:
            Parameter: the parameter passed in.
        """
        if parameter is None:
            self._parameters[name] = None
        elif not isinstance(parameter, framework.Parameter):
            raise TypeError(
                "parameter assignment requires Parameter or None, but got '{}'"
                .format(type(parameter).__name__))

        if len(self._loaddict_holder) > 0:
            assert parameter.name in self._loaddict_holder, "Parameter not found, Can't not find [ {} ] in stat_dict".format(
                parameter.name)

            parameter.set_value(self._loaddict_holder[parameter.name])

        self._parameters[name] = parameter
        return parameter

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._sub_layers:
            return self._sub_layers[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        def _remove_if_exist(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
        params = self.__dict__.get('_parameters', None)
        if isinstance(value, framework.Parameter):
            if params is None:
                raise ValueError(
                    "super(YourLayer, self).__init__() should be called first")
            if len(self._loaddict_holder) > 0:
                assert value.name in self._loaddict_holder, "Parameter not found, Can't not find [ {} ] in stat_dict".format(
                    value.name)

                value.set_value(self._loaddict_holder[value.name])

            _remove_if_exist(self.__dict__, self._sub_layers)
            params[name] = value
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    "assignment to parameter '{}' should be of type Parameter or None, but got '{}'"
                    .format(name, type(value).__name__))
            params[name] = None
        else:
            layers = self.__dict__.get('_sub_layers', None)
            if isinstance(value, core.Layer):
                if layers is None:
                    raise ValueError(
                        "super(YourLayer, self).__init__() should be called first"
                    )

                _remove_if_exist(self.__dict__, self._parameters)
                layers[name] = value
            elif layers is not None and name in layers:
                if value is not None:
                    raise TypeError(
                        "assignment to sublayer '{}' should be of type Layer or None, but got '{}'"
                        .format(name, type(value).__name__))
                layers[name] = None
            else:
                object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._sub_layers:
            del self._sub_layers[name]
        else:
            object.__delattr__(self, name)

    def state_dict(self,
                   destination=None,
                   include_sublayers=True,
                   structured_name_prefix=""):
        '''
        Get all parameters of current layer and its sub-layers. And set all the parameters into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters will set to this dict . Default: None
            include_sublayers(bool, optional) : If true, also include the parameters from sublayers. Default: True

        Retruns:
            dict: a dict contains all the parameters

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])

                    state_dict = emb.state_dict()
                    fluid.save_dygraph( state_dict, "paddle_dy")

        '''

        if destination is None:
            destination = collections.OrderedDict()
        for name, data in self._parameters.items():
            if data is not None:
                destination[structured_name_prefix + name] = data

        if include_sublayers:
            for layer_name, layer_item in self._sub_layers.items():
                if layer_item is not None:
                    destination_temp = destination.copy()
                    destination_temp.update(
                        layer_item.state_dict(
                            destination_temp, include_sublayers,
                            structured_name_prefix + layer_name + "."))
                    destination = destination_temp
        return destination

    def set_dict(self,
                 stat_dict,
                 include_sublayers=True,
                 use_structured_name=True):
        '''
        Set parameters from stat_dict. All the parameters will be reset by the tensor in the stat_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters
            include_sublayers(bool, optional) : If true, also include the parameters from sublayers. Default: True
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter name as key. 
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])

                    state_dict = emb.state_dict()
                    fluid.save_dygraph( state_dict, "paddle_dy")
                    
                    para_state_dict, _ = fluid.load_dygraph( "paddle_dy")

                    emb.set_dict( para_state_dict )

        '''
        self.load_dict(
            stat_dict,
            include_sublayers=include_sublayers,
            use_structured_name=use_structured_name)

    def load_dict(self,
                  stat_dict,
                  include_sublayers=True,
                  use_structured_name=True):
        '''
        Set parameters from stat_dict. All the parameters will be reset by the tensor in the stat_dict

        This api will be Deprecated. Please use set_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters
            include_sublayers(bool, optional) : If true, also include the parameters from sublayers. Default: True
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter name as key.
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])

                    state_dict = emb.state_dict()
                    fluid.save_dygraph( state_dict, "paddle_dy")
                    
                    para_state_dict, _ = fluid.load_dygraph( "paddle_dy")

                    emb.load_dict( para_state_dict )

        '''

        inner_state_dict = self.state_dict()

        for name, para in inner_state_dict.items():
            key_name = name if use_structured_name else para.name
            if key_name in stat_dict:
                para.set_value(stat_dict[key_name])
            else:
                raise RuntimeError(
                    "Parameter not found, Can't not find [ {} ] in stat_dict"
                    "use_structured_name is set to [{}]".format(
                        key_name, use_structured_name))
        unused_para_list = []
        for k, v in stat_dict.items():
            if k not in inner_state_dict:
                unused_para_list.append(k)
        if len(unused_para_list) > 0:
            warnings.warn(
                "Varibale [ {} ] are not used, because not included in layers state_dict".
                format(" ".join(unused_para_list)))
