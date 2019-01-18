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
from ..graph import PyGraph


class QuantizationPerformer(object):
    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 activation_quantize_type='abs_max',
                 weight_quantize_type='abs_max',
                 window_size=10000):
        """
        Convert and rewrite the IRGraph according to weight and
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
            graph = IRGraph(program)
            performer = QuantizationPerformer()
            performer.quantize_transform(graph)
        """
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

        quant_type = ['abs_max', 'range_abs_max']
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be ",
                "'abs_max' or 'range_abs_max'.", str(activation_quantize_type))
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be ",
                "'abs_max' or 'range_abs_max'.", str(weight_quantize_type))

        self.activation_quantize_type = activation_quantize_type
        self.weight_quantize_type = weight_quantize_type
        self.window_size = window_size

        self.need_inited_outer = collections.OrderedDict()
        self.quantizable_ops = ['conv2d', 'depthwise_conv2d', 'mul']
        self.quantizable_grad_ops = [
            '%s_grad' % (op) for op in self.quantizable_ops
        ]
        self.fake_quant_op_types = [
            'fake_quantize_abs_max', 'fake_quantize_range_abs_max'
        ]
        self.fake_dequant_op_types = ['fake_dequantize_max_abs']
        self.is_test = None
        self.global_step = None

    def quantize_transform(self, graph, is_test):
        self.need_inited_outer.clear()
        self.is_test = is_test
        assert isinstance(graph,
                          PyGraph), 'graph must be the instance of PyGraph.'
        # marked the variable which has been dequantized.
        dequantized_vars = collections.OrderedDict()
        params = [p.name() for p in graph.all_parameters()]

        def _transform_forward(graph, op):
            for var_node in op.inputs:
                if var_node.name() in dequantized_vars:
                    dequant_var_node = dequantized_vars[var_node.name()]
                else:
                    quant_bits = self.weight_bits if var_node.name() in params \
                    else self.activation_bits
                    quant_type = self.weight_quantize_type if var_node.name() \
                        in params else self.activation_quantize_type
                    quant_var_node, scale_var_node = self._insert_quant_op(
                        graph, var_node, quant_bits, quant_type)
                    dequant_var_node = self._insert_dequant_op(
                        graph, quant_var_node, scale_var_node, quant_bits)
                    dequantized_vars[var_node.name()] = dequant_var_node
                self._update_input(var_node, dequant_var_node, op)

        if not self.is_test:
            self._create_global_step(graph)
        ops = graph.all_ops()
        for op in ops:
            # transform the forward graph
            if op.name() in self.quantizable_ops:
                _transform_forward(graph, op)
            # rename the inputs of backward op
            if op.name() in self.quantizable_grad_ops:
                _transform_backward(graph, op)
        return self.need_inited_outer

    def _insert_quant_op(self, graph, var_node, quant_bits, quant_type):
        """
        Insert fake_quantize_op in the graph.
        """
        if quant_type == 'abs_max':
            return self._insert_quant_abs_max_op(graph, var_node, quant_bits)
        elif quant_type == 'range_abs_max':
            return self._inser_quant_range_abs_max_op(graph, var_node,
                                                      quant_bits)

    def _insert_quant_abs_max_op(self, graph, var_node, quant_bits):
        """
        Insert fake_quantize_abs_max op in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(var_node.name()),
            var_type=var_node.var().type(),
            shape=var_node.var().shape(),
            var_dtype=var_node.var().dtype())
        scale_var_node = graph.create_var_node(
            name=self._quantized_scale_name(var_node.name()),
            var_type=var_node.var().type(),
            shape=var_node.var().shape(),
            var_dtype=var_node.var().dtype())
        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_abs_max',
            attrs={'bit_length': quant_bits},
            inputs={'X': var_node},
            outputs={'Out': quant_var_node,
                     'OutScale': scale_var_node})
        self._link_to(var_node, quant_op_node)
        self._link_to(quant_op_node, quant_var_node)
        self._link_to(quant_op_node, scale_var_node)
        return quant_var_node, scale_var_node

    def _insert_quant_range_abs_max_op(self, graph, var_node, quant_bits):
        """
        Insert fake_quantize_range_abs_max on the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(var_node.name()),
            var_type=var_node.var().type(),
            shape=var_node.var().shape(),
            var_dtype=var_node.var().dtype())

        scale_in_node = graph.create_param_node(
            name=self._quantized_scale_name(var_node.name()),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[1],
            var_dtype=var_node.var().dtype())
        self.need_inited_outer[scale_in_node.var()] = Constant(value=0.001)

        scale_out_node = graph.create_var_node_from_desc(scale_in_node.var())
        inputs = {'X': var_node, 'InScale': scale_in_node}
        outputs = {'Out': quant_var_node, 'OutScale': scale_out_node}

        if not self.is_test:
            # The name of scales_var_node maybe 'scales_0', 'scales_1', etc.
            scales_node = graph.create_param_node(
                name=unique_name.generate('scales'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                shape=[self.window_size],
                var_dtype=var_node.var().dtype())
            self.need_inited_outer[scales_node.var()] = Constant(value=0)
            inputs['Iter'] = self.global_step
            outputs['OutScales'] = scales_node
        attrs = {
            'window_size': self.window_size,
            'bit_length': quant_bits,
            'is_test': self.is_test
        }
        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_range_abs_max',
            attrs=attrs,
            inputs=inputs,
            outputs=outputs)

        self._link_to(var_node, quant_op_node)
        self._link_to(scale_in_node, quant_op_node)
        self._link_to(quant_op_node, quant_var_node)
        self._link_to(quant_op_node, scale_out_node)

        if not self.is_test:
            self._link_to(self.global_step, quant_op_node)
            self._link_to(quant_op_node, scales_node)

        return quant_var_node, scale_out_node

    def _insert_dequant_op(self, graph, var_node, scale_var_node, quant_bits):
        """
        Insert fake_dequantize_op in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(var_node.name()),
            var_type=var_node.var().type(),
            shape=var_node.var().shape(),
            var_dtype=var_node.var().dtype())
        max_range = (1 << (quant_bits - 1)) - 1
        dequant_op_node = graph.create_op_node(
            op_type='fake_dequantize_max_abs',
            attrs={'max_range': float(max_range)},
            inputs={'X': var_node,
                    'Scale': scale_var_node},
            outputs={'Out': dequant_var_node})
        self._link_to(var_node, dequant_op_node)
        self._link_to(scale_var_node, dequant_op_node)
        self._link_to(dequant_op_node, dequant_var_node)
        return dequant_var_node

    def _update_input(self, old_input_node, new_input_node, op_node):
        old_input_node.outputs.remove(op_node)
        op_node.inputs.remove(old_input_node)
        new_input_node.outputs.append(op_node)
        op_node.inputs.append(new_input_node)

    def _link_to(node_in, node_out):
        node_in.outputs.append(node_out)
        node_out.inputs.append(node_in)

    def _quantized_var_name(self, var_name):
        """
        Return quantized variable name for the input `var_name`.
        """
        return "%s.quantized" % (var_name)

    def _dequantized_var_name(self, var_name):
        """
        Return dequantized variable name for the input `var_name`.
        """
        return "%s.dequantized" % (var_name)

    def _quantized_scale_name(self, var_name):
        """
        Return quantized variable name for the input `var_name`.
        """
        return "%s.scale" % (var_name)

    def _original_var_name(self, var_name):
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

    def _is_float(self, v):
        return isinstance(v, float) or isinstance(v, np.float32)

    def _quant(self, x, scale, num_bits):
        y = np.round(x / scale * ((1 << (num_bits - 1)) - 1))
        return y
