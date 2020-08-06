# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import six
import numpy as np
import collections
from paddle.fluid import core
from paddle.fluid.dygraph import layers
from paddle.fluid.layers.utils import flatten
from paddle.fluid.layers.utils import pack_sequence_as
from paddle.fluid.framework import convert_np_dtype_to_dtype_, Variable
from paddle.fluid.dygraph.base import switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.utils import type_name
from paddle.fluid.dygraph.dygraph_to_static.utils import parse_arg_and_kwargs


class TensorSpec(object):

    __slots__ = ['_shape', '_dtype', '_name']

    def __init__(self, shape, dtype='float32', name=None):
        # replace `None` in shape  with -1
        self._shape = self._verify(shape)
        # convert dtype into united represention
        if dtype is not None:
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)
        self._dtype = dtype
        self._name = name

    @classmethod
    def from_variable(cls, variable, name=None):
        if isinstance(variable, (Variable, core.VarBase)):
            print('here')
            return cls(variable.shape, variable.dtype, name or variable.name)
        else:
            raise ValueError(
                "Input `variable` should be a Variable, but received {}.".
                format(type_name(variable)))

    @classmethod
    def from_numpy(cls, ndarray, name=None):
        # TODO: tansform ndarray.type into paddle dataType
        return cls(ndarray.shape, ndarray.dtype, name)

    # TODO: where to use this interface?
    def batch(self, batch_size):
        """
        Insert `batch_size` in front of the `shape`.
        """
        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) != 1:
                raise ValueError(
                    "Length of  {}: batch_size shall be 1, but received {}.".
                    format(type_name(variable), len(batch_size)))
            batch_size = batch_size[1]
        elif not isinstance(batch_size, (int, long)):
            raise TypeError(
                "type(batch_size) shall be int or long, but received {}.".
                format(type_name(batch_size)))

        new_shape = [batch_size] + list(self._shape)
        return TensorSpec(new_shape, self._dtype, self._name)
        # TODO: or whether consider to return self?

    def unbatch(self):
        if len(self._shape) == 0:
            raise ValueError(
                "Not support to unbatch a variable when len(shape) == 0.")

        return TensorSpec(self._shape[1:], self._dtype, self.name)

    def _verify(self, shape):
        if not isinstance(shape, (list, tuple)):
            raise TypeError(
                "Type of `shape` in TensorSpec should be one of (tuple, list), but received {}.".
                format(type_name(shape)))
        if len(shape) == 0:
            raise ValueError(
                "`shape` in TensorSpec should contains at least 1 element, but received empty shape."
            )

        for i, ele in enumerate(shape):
            if ele is not None:
                if not isinstance(ele, (int, long)):
                    raise ValueError(
                        "shape[{}] should be a int, but received `{}`:{}.".
                        format(i, type_name(ele), ele))
            if ele is None or ele < -1:
                shape[i] = -1

        return tuple(shape)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return '{}(shape={}, dtype={}, name={})'.format(
            type_name(self), self._shape, self._dtype, self._name)

    def __hash__(self):
        return hash((self._shape, self._dtype))

    def __eq__(self, other):
        return (type(self) is type(other) and all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__slots__))

    def __ne__(self, other):
        return not self == other


def get_parameters(layer_instance, include_sublayer=True):
    """
    Returns parameters of decorated layers. If set `include_sublayer` True,
    the parameters created in sub layers will be added.
    """
    params = collections.OrderedDict()
    if layer_instance is not None:
        if isinstance(layer_instance, layers.Layer):
            if include_sublayer:
                params = layer_instance.parameters()
                names = [p.name for p in params]
                params = collections.OrderedDict(zip(names, params))
            else:
                params = layer_instance._parameters
        else:
            raise TypeError(
                "Type of `layer_instance` should be layer.Layer, but received {}".
                format(type_name(layer_instance)))

    return params


def get_buffers(layer_instance, include_sublayer=True):
    """
    Returns Variable buffers of decorated layers. If set `include_sublayer` True,
    the Variable buffers created in sub layers will be added.
    """
    buffers = collections.OrderedDict()
    if layer_instance is not None:
        if isinstance(layer_instance, layers.Layer):
            if include_sublayer:
                buffers = layer_instance.buffers()
                names = [buffer.name for buffer in buffers]
                buffers = collections.OrderedDict(zip(names, buffers))
            else:
                buffers = layer_instance._buffers
        else:
            raise TypeError(
                "Type of `layer_instance` should be layer.Layer, but received {}".
                format(type_name(layer_instance)))
    return buffers


class FunctionSpec(object):
    # TODO: consider jit_save and whether we need args and kwargs
    def __init__(self, function, input_signature=None, is_method=False):
        self._dyfunc = function
        if input_signature is None:
            self._input_signature = None
            self._flatten_signature = None
        else:
            if not isinstance(input_signature, (tuple, list)):
                raise TypeError(
                    "Type of `input_signature` should be one of (tuple, list), but received {}.".
                    format(type_name(input_signature)))
            self._input_signature = tuple(input_signature)
            self._flatten_signature = flatten(self._input_signature)

        self._is_method = is_method
        # parse full argument names list.
        self._arg_names, self._default_kwargs = parse_arg_and_kwargs(function)

        self._idx_to_variable_spec = {}
        # self._inputs_with_spec = self.args_to_variable_spec(args, kwargs)

    def unified_args_and_kwargs(self, args, kwargs):
        # TODO: move kwargs default value into args
        if len(self._arg_names) < len(args):
            raise ValueError(
                "The decorated function `{}` requires {} arguments, but received {}.".
                format(self._dyfunc.__name__, len(self._arg_names), len(args)))

        args = list(args)
        for i in six.moves.range(len(args), len(self._arg_names)):
            arg_name = self._arg_names[i]
            if arg_name in kwargs:
                args.append(kwargs[arg_name])
                del kwargs[arg_name]
            else:
                args.append(self._default_kwargs[arg_name])

        return tuple(args), kwargs

    def args_to_variable_spec(self, args, kwargs):
        """
        Convert input arguments into TensorSpec.
        
        1. If specific input_signature, use them to construct feed layers.
        2. If input_signature is None, consider all Tensor and Numpy.ndarray as feed layers
        """
        inputs_with_spec = []
        flat_args = flatten(args)

        if self._input_signature is not None:
            if kwargs:
                raise ValueError("Not support kwargs when specific TensorSpec.")
            # TODO: consider nested structure input arguments
            flat_signature = flatten(self._input_signature)

            if len(flat_args) < len(flat_signature):
                raise ValueError(
                    "Mismatch length of arguments and TensorSpec, receive len(args):{} < len(TensorSpec): {}".
                    format(len(args), len(flat_signature)))

            # TODO: make sure we can handle When the lengths are inconsistent
            # print(args)

            inputs_with_spec = flat_signature + list(flat_args)[len(
                flat_signature):]

            self._idx_to_variable_spec = dict(
                zip(range(len(flat_signature)), flat_signature))
            # print(inputs_with_spec)
        else:
            # map index into variable_spec
            for idx, input_var in enumerate(flat_args):
                if isinstance(input_var, np.ndarray):
                    input_var = TensorSpec.from_numpy(input_var)
                elif isinstance(input_var, core.VarBase):
                    input_var = TensorSpec.from_variable(input_var)

                if isinstance(input_var, TensorSpec):
                    self._idx_to_variable_spec[idx] = input_var

                inputs_with_spec.append(input_var)

        return inputs_with_spec

    # TODO: Reuse code with to_static_inputs
    @switch_to_static_graph
    def to_static_inputs_with_signature(self, inputs_with_spec, main_program):
        """
        If users specific signature info, we only signature?
        """
        flat_inputs_spec = flatten(inputs_with_spec)

        inputs = []
        block = main_program.global_block()
        for i, var_spec in enumerate(flat_inputs_spec):
            if isinstance(var_spec, TensorSpec):
                feed_layer = block.create_var(
                    # TODO: consider more elegant way to name this
                    name=var_spec.name or "feed_%s" % i,
                    shape=var_spec.shape,
                    dtype=var_spec.dtype,
                    is_data=True,
                    need_check_feed=False)
            else:
                feed_layer = var_spec
            inputs.append(feed_layer)

        return pack_sequence_as(inputs_with_spec, inputs)

    @switch_to_static_graph
    def to_static_inputs(self, main_program):
        inputs = []
        block = main_program.global_block()
        for input_var in flatten(self.args):
            if isinstance(input_var, np.ndarray):
                feed_layer = block.create_var(
                    name=unique_name.generate('feed'),
                    shape=list(input_var.shape),
                    dtype=input_var.dtype,
                    is_data=True,
                    need_check_feed=False)
            elif isinstance(input_var, core.VarBase):
                feed_layer = block.create_var(
                    name=input_var.name,
                    shape=list(input_var.shape),
                    dtype=input_var.dtype,
                    stop_gradient=input_var.stop_gradient,
                    need_check_feed=False)
            else:
                feed_layer = input_var

            inputs.append(feed_layer)
        # Restores the nested structure as self.args
        return pack_sequence_as(self.args, inputs)

    @property
    def dyfunc(self):
        return self._dyfunc

    @property
    def args(self):
        return self._args

    @property
    def is_method(self):
        return self._is_method
        # return self._args and isinstance(self._args[0], layers.Layer)

    # TODO: consider input_signature as key

    def __key(self):
        # Note: if dygraph function is a method of class,
        # consider instance info as hash key.
        if self.is_method():
            # NOTE: we can use Layer's (instance + function code) as hash key.
            # An instance will not hold two identical methods 
            return self._dyfunc_code, self._args[0], tuple(
                self._inputs_with_spec)
        else:
            return self._dyfunc, tuple(self._inputs_with_spec)

    # def __hash__(self):
    #     return hash(self.__key())

    @property
    def hash_keys(self):
        return self.__key()

    @property
    def code(self):
        dyfunc = getattr(self._function, '__wrapped__', self._function)
        return inspect.getsource(dyfunc)
