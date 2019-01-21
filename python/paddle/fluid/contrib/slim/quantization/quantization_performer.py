#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from .... import core
from ....initializer import Constant
from .... import unique_name
from ....framework import Variable

_QUANTIZABLE_OP_TYPES = ['conv2d', 'depthwise_conv2d', 'mul']

_NEED_INITIALIZED_VARS_OR_PARAMS = collections.OrderedDict()


def _quantized_var_name(var_name):
    """
    Return quantized variable name for the input `var_name`.
    """
    return "%s.quantized" % (var_name)


def _dequantized_var_name(var_name):
    """
    Return dequantized variable name for the input `var_name`.
    """
    return "%s.dequantized" % (var_name)


def _quantized_scale_name(var_name):
    """
    Return quantized variable name for the input `var_name`.
    """
    return "%s.scale" % (var_name)


def _original_var_name(var_name):
    """
    Return the original variable name.
    """
    if var_name.endswith('.quantized.dequantized'):
        return var_name[:-len('.quantized.dequantized')]
    if var_name.endswith('.quantized'):
        return var_name[:-len('.quantized')]
    if var_name.endswith('.dequantized'):
        return var_name[:-len('.dequantized')]
    if var_name.endswith('.scale'):
        return var_name[:-len('.scale')]
    else:
        return var_name


def _is_float(v):
    return isinstance(v, float) or isinstance(v, np.float32)


def quant(x, scale, num_bits):
    y = np.round(x / scale * ((1 << (num_bits - 1)) - 1))
    return y


class QuantizationPerformer(object):
    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 activation_quantize_type='abs_max',
                 weight_quantize_type='abs_max',
                 window_size=10000):
        """
        Convert and rewrite the IR Graph according to weight and
        activation quantization type.

        Args:
            weight_bits (int): quantization bit number for weights,
                the bias is not quantized.
            activation_bits (int): quantization bit number for activation.
            activation_quantize_type (str): quantization type for activation,
                now support 'abs_max', 'range_abs_max'. If use 'abs_max' mode,
                the quantization scale will be calculated dynamically each step
                in both training and testing period. If use 'range_abs_max',
                a static quantization scale will be calculated during training
                and used in inference.
            weight_quantize_type (str): quantization type for weights,
                support 'abs_max'. The 'range_abs_max' usually is not used for
                weight, since weights are fixed once the model is well trained.
            window_size (int): the window size for 'range_abs_max' quantization.

        Examples:

        .. code-block:: python

            # the original graph will be rewrite, if you don't want to
            # change it, please clone at first.
            # graph = graph.clone()
            from paddle.fluid.contrib.slim import *
            from paddle.fluid.contrib.quantize import *

            graph = ImitationGraph(program)
            t = QuantizationPerformer()

            t.quantize_transform(graph)
        """
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        quant_type = ['abs_max', 'range_abs_max']
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be ",
                "'abs_max' or 'range_abs_max'.", str(weight_quantize_type))
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be ",
                "'abs_max' or 'range_abs_max'.", str(activation_quantize_type))

        self.weight_quantize_type = weight_quantize_type
        self.activation_quantize_type = activation_quantize_type

        self.window_size = window_size
        self.fake_quant_op_types = [
            'fake_quantize_abs_max', 'fake_quantize_range_abs_max'
        ]
        self.fake_dequant_op_types = ['fake_dequantize_max_abs']
        self.is_test = None
        self.global_step = None

    def _create_global_step(self, graph):
        if self.weight_quantize_type == 'range_abs_max' or \
                self.activation_quantize_type == 'range_abs_max':
            counter_name = '@STEP_COUNTER@'
            if counter_name not in graph.vars_map():
                counter = graph.create_var(
                    name=counter_name,
                    dtype='int64',
                    shape=[1],
                    persistable=True)
                _NEED_INITIALIZED_VARS_OR_PARAMS[counter] = Constant(
                    value=0, force_cpu=True)
                graph.prepend_op(
                    type='increment',
                    inputs={'X': [counter]},
                    outputs={'Out': [counter]},
                    attrs={'step': 1.0})
                counter.stop_gradient = True
            self.global_step = graph.var(counter_name)

    def _insert_quant_abs_max_op(self, graph, idx, var, quant_bits):
        """Insert fake_quantize_abs_max op.
        """
        quant_var = graph.create_var(
            name=_quantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        scale = graph.create_var(
            name=_quantized_scale_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        quant_op = graph.insert_op(
            idx,
            type='fake_quantize_abs_max',
            attrs={'bit_length': quant_bits},
            inputs={'X': var},
            outputs={'Out': quant_var,
                     'OutScale': scale})
        return quant_var, scale

    def _insert_quant_range_abs_max_op(self, graph, idx, var, quant_bits):
        """Insert fake_quantize_range_abs_max
        """
        quant_var = graph.create_var(
            name=_quantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        scale = graph.create_parameter(
            name=_quantized_scale_name(var.name),
            shape=[1],
            dtype=var.dtype,
            trainable=False)
        scale.stop_gradient = True
        _NEED_INITIALIZED_VARS_OR_PARAMS[scale] = Constant(value=0.001)
        ins = {'X': var, 'InScale': scale}
        outs = {'Out': quant_var, 'OutScale': scale}
        if not self.is_test:
            scales = graph.create_var(
                name=unique_name.generate('scales'),
                persistable=True,
                dtype=var.dtype,
                shape=[self.window_size])
            _NEED_INITIALIZED_VARS_OR_PARAMS[scales] = Constant(value=0)
            # A global step counter variable with type int64
            ins['Iter'] = self.global_step
            outs['OutScales'] = scales

        attrs = {
            'window_size': self.window_size,
            'bit_length': quant_bits,
            'is_test': self.is_test
        }

        quant_op = graph.insert_op(
            idx,
            type='fake_quantize_range_abs_max',
            attrs=attrs,
            inputs=ins,
            outputs=outs)

        return quant_var, scale

    def _insert_quant_op(self, graph, idx, var, quant_bits, quant_type):
        """
        Insert fake_quantize_op
        """
        if quant_type == 'abs_max':
            return self._insert_quant_abs_max_op(graph, idx, var, quant_bits)
        elif quant_type == 'range_abs_max':
            return self._insert_quant_range_abs_max_op(graph, idx, var,
                                                       quant_bits)

    def _insert_dequant_op(self, graph, idx, var, scale, quant_bits):
        """
        Insert fake_quantize_op
        """
        dequant_var = graph.create_var(
            name=_dequantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        # insert fake_dequantize_op
        max_range = (1 << (quant_bits - 1)) - 1
        dequant_op = graph.insert_op(
            idx,
            type="fake_dequantize_max_abs",
            attrs={'max_range': float(max_range)},
            inputs={"X": var,
                    'Scale': scale},
            outputs={"Out": dequant_var})
        return dequant_var

    def quantize_transform(self, graph, place):
        _NEED_INITIALIZED_VARS_OR_PARAMS.clear()
        self.is_test = False
        if graph is None:
            raise ValueError("The graph cannot be None!")
        # marked the variable which has been quantized and dequantized.
        dequanted_vars = collections.OrderedDict()
        grad_op_types = ['%s_grad' % (type) for type in _QUANTIZABLE_OP_TYPES]
        params = [p.name for p in graph.all_parameters()]

        def _transform_forward(graph, op):
            idx = graph.index(op)
            # insert quant op and dequant op
            for name in op.input_arg_names:
                if name in dequanted_vars:
                    dequant_var = dequanted_vars[name]
                else:
                    var = graph.var(name)
                    quant_bits = self.weight_bits if var.name in params \
                        else self.activation_bits
                    quant_type = self.weight_quantize_type if var.name \
                                                              in params else self.activation_quantize_type

                    quant_var, scale_var = self._insert_quant_op(
                        graph, idx, var, quant_bits, quant_type)
                    dequant_var = self._insert_dequant_op(
                        graph, idx + 1, quant_var, scale_var, quant_bits)
                    dequanted_vars[name] = dequant_var
                # rename the forward op inputs
                op._rename_input(name, dequant_var.name)

        def _transform_backward(graph, op):
            no_dequanted_input_vars = True
            for name in op.input_arg_names:
                if name in dequanted_vars:
                    dequant_var = dequanted_vars[name]
                    op._rename_input(name, dequant_var.name)
                    no_dequanted_input_vars = False
            if no_dequanted_input_vars:
                raise ValueError("There is no dequanted inputs for op %s." %
                                 (op.type))

        self._create_global_step(graph)
        for op in list(graph.all_ops()):
            # rewrite the forward graph
            if op.type in _QUANTIZABLE_OP_TYPES:
                _transform_forward(graph, op)
            # rename the backward op inputs
            if op.type in grad_op_types:
                _transform_backward(graph, op)
        graph.init_vars(_NEED_INITIALIZED_VARS_OR_PARAMS, place)

    def freeze_graph(self, graph, place, fuse_bn=False):
        """
        Freeze input graph for inference.

        Args:
            graph (Graph): the input graph to be freeze.
        """
        if graph is None:
            raise ValueError("The graph cannot be None!")

        self.is_test = True

        if fuse_bn:
            # TODO: BNFuseQuantizationPerformer
            pass

        persistable_vars = [
            v.name
            for v in filter(lambda var: var.persistable, graph.all_vars())
        ]
        op_in_rename_map = collections.OrderedDict()
        op_out_rename_map = collections.OrderedDict()
        var_scale_map = collections.OrderedDict()

        def _remove_fake_quant_and_dequant_op(graph, op):
            idx = graph.index(op)
            k = op.output('Out')[0]
            v = op.input('X')[0]
            if v not in op_in_rename_map:
                op_in_rename_map[k] = v
            else:
                op_in_rename_map[k] = op_in_rename_map[v]
            graph.remove_op(idx)

        def _insert_post_dequant_op(graph, op):
            idx = graph.index(op)
            max_range = None
            scale_var = None
            for name in op.input_arg_names:
                if name in op_in_rename_map:
                    op._rename_input(name, op_in_rename_map[name])

                scale_v = var_scale_map[_original_var_name(name)]
                if _original_var_name(name) in persistable_vars:
                    param_range = (1 << (self.weight_bits - 1)) - 1
                    act_range = (1 << (self.activation_bits - 1)) - 1
                    assert _is_float(scale_v)
                    max_range = param_range * act_range / scale_v
                else:
                    assert isinstance(scale_v, Variable)
                    scale_var = var_scale_map[_original_var_name(name)]

            if len(op.output_arg_names) != 1:
                raise ValueError("Only support one output, but op %s has"
                                 " more than one output." % (op.type))
            out_var = graph.var(op.output_arg_names[0])
            dequant_var = graph.create_var(
                name=_dequantized_var_name(out_var.name),
                type=out_var.type,
                shape=out_var.shape,
                dtype=out_var.dtype)
            # insert fake_dequantize_op
            dequant_op = graph.insert_op(
                idx + 1,
                type="fake_dequantize_max_abs",
                attrs={'max_range': float(max_range)},
                inputs={"X": out_var,
                        'Scale': scale_var},
                outputs={"Out": dequant_var})
            op_out_rename_map[out_var.name] = dequant_var.name
            return dequant_var

        def _load_var(name):
            return np.array(graph.scope.find_var(name).get_tensor())

        def _restore_var(name, arr):
            t = graph.scope.find_var(name).get_tensor()
            t.set(arr, place)

        for op in list(graph.all_ops()):
            op_type = op.type

            # insert dequant_op after fc/conv, need to rename
            # input of the followed ops
            for name in op.input_arg_names:
                if name in op_out_rename_map:
                    op._rename_input(name, op_out_rename_map[name])

            if op_type in self.fake_quant_op_types:
                in_arg_name = op.input('X')[0]
                if in_arg_name in persistable_vars:
                    if self.weight_quantize_type == 'abs_max':
                        param = _load_var(in_arg_name)
                        scale_v = np.max(np.abs(param))
                    else:
                        scale_v = _load_var(op.output('OutScale')[0])
                    var_scale_map[in_arg_name] = scale_v
                else:
                    scale_v = graph.var(op.output('OutScale')[0])
                    var_scale_map[in_arg_name] = scale_v

                if in_arg_name in persistable_vars:
                    _remove_fake_quant_and_dequant_op(graph, op)
                    # quantize weight and restore
                    param_t = _load_var(in_arg_name)
                    param_q_t = quant(param_t, scale_v, self.weight_bits)
                    _restore_var(in_arg_name, param_q_t)

            if op_type in self.fake_dequant_op_types:
                _remove_fake_quant_and_dequant_op(graph, op)

            if op_type in _QUANTIZABLE_OP_TYPES:
                dequant_var = _insert_post_dequant_op(graph, op)

        # remove the unused var in graph
        self._remove_unused_var(graph)

    def _remove_unused_var(self, graph):
        args = []
        for op in list(graph.all_ops()):
            args += op.input_arg_names
            args += op.output_arg_names
        args = list(set(args))
        var_names = graph.vars_map().keys()
        remove_vars = []
        for var in var_names:
            if var not in args:
                remove_vars.append(var)

        for v in remove_vars:
            graph.remove_var(v)

    def convert_to_int8(self, graph, place):
        """
        Covert input graph into quantized int8 graph.
        """
        if graph is None:
            raise ValueError("The graph cannot be None!")

        def _load_var(name):
            return np.array(graph.scope.find_var(name).get_tensor())

        def _convert_to_int8(var):
            int8_var_name = var.name + ".int8"
            int8_var = graph.create_parameter(
                name=int8_var_name.encode('ascii'),
                type=var.type,
                dtype=core.VarDesc.VarType.INT8,
                shape=var.shape)

            tensor = _load_var(var.name)

            graph.scope.var(int8_var_name)

            int8_tensor = graph.scope.find_var(int8_var_name).get_tensor()
            int8_tensor.set(tensor.astype(np.int8), place)
            return int8_var

        input_map = {}
        for op in list(graph.all_ops()):
            if op.type in _QUANTIZABLE_OP_TYPES:
                for name in op.input_arg_names:
                    var = graph.var(name)
                    if var.persistable:
                        if name not in input_map:
                            int8_var = _convert_to_int8(var)
                            input_map[name] = int8_var.name
                        op._rename_input(name, input_map[name])
        self._remove_unused_var(graph)
