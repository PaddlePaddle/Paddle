#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import IrGraph, Variable, Program
import paddle


class TransformForMkldnnPass(object):
    def __init__(self, scope=None, place=None):
        """
        Convert IrGraph to MKL-DNN suppport INT8 runnable graph according 
        to weight and activation quantization type.
    
        Args:
            scope(fluid.Scope): When activation use 'range_abs_max' as the quantize
            type, this pass will create some new parameters. The scope is used to
            initialize these new parameters.
            place(fluid.CPUPlace|fluid.CUDAPlace): place is used to initialize new
            parameters described above.

        """

        self._scope = scope
        self._place = place

        self.quantize_type = [
            'fake_quantize_moving_average_abs_max',
            'fake_quantize_range_abs_max'
        ]
        self.dequantize_type = ['fake_dequantize_max_abs']

        self._quantizable_ops = ['conv2d', 'depthwise_conv2d', 'mul']
        self._conv_ops = ['conv2d', 'depthwise_conv2d']

        self.InScale = {}
        self.max_range = {}
        self.conv_new_output = {}
        self.s8_max = 127
        # Temporary for keeping the mul op fake quantization
        self.mul_input_id = []
        self.mul_output_id = []

    def apply(self, graph):
        """
        Quantize the graph for MKL-DNN inference process. According to 
        weight and activation quantization type, the graph will transform 
        fake quantize operators to quantize operators and remove the fake 
        dequantize operators.
      
        Args:
            graph(IrGraph): the applied graph.
        """

        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        ops = graph.all_op_nodes()

        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        #Collect the InScales and max_range to calculate the new scales for MKL-DNN INT8 conv
        for op_node in ops:
            if op_node.name() in self.dequantize_type:
                input_name = op_node.input("X")[0]
                scale_name = op_node.input("Scale")[0]
                self.InScale[input_name] = self._load_param(self._scope,
                                                            scale_name)[0]
                self.max_range[input_name] = op_node.op().attr("max_range")
                self.conv_new_output[input_name] = op_node.output("Out")[0]
            # Temporary graph transform on keeping the mul
            elif op_node.name() in ['mul']:
                input_node = graph._find_node_by_name(op_node.inputs,
                                                      op_node.input('X')[0])
                output_node = graph._find_node_by_name(op_node.outputs,
                                                       op_node.output('Out')[0])
                self.mul_input_id.append(input_node.id())
                self.mul_output_id.append(output_node.id())

        for op_node in ops:
            if op_node.name() in self._conv_ops:
                self._transform_to_conv_mkldnn(graph, op_node)
            elif op_node.name() in self.quantize_type:
                self._transform_to_quantize_mkldnn(graph, op_node)
            elif op_node.name() in self.dequantize_type:
                self._remove_fake_dequantize_op(graph, op_node)
            self._remove_unused_var_nodes(graph)
        return graph

    def _transform_to_conv_mkldnn(self, graph, op_node):
        weight_name = op_node.input("Filter")[0]
        output_name = op_node.output("Output")[0]
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(
            np.multiply(weight, 127),
            self.max_range[op_node.output("Output")[0]])
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)
        input_var_node = graph._find_node_by_name(op_node.inputs,
                                                  op_node.input("Input")[0])
        weight_var_node = graph._find_node_by_name(op_node.inputs,
                                                   op_node.input("Filter")[0])

        output_var_node = graph._find_node_by_name(
            graph.all_var_nodes(),
            self.conv_new_output[op_node.output("Output")[0]])
        attrs = {
            name: op_node.op().attr(name)
            for name in op_node.op().attr_names()
        }

        conv_op_node = graph.create_op_node(
            op_type='conv2d',
            attrs=attrs,
            inputs={'Input': input_var_node,
                    'Filter': weight_var_node},
            outputs={'Output': output_var_node})

        # Based on the QAT scale value to calculate the scales for MKL-DNN INT8 conv2d
        scale_in = self.s8_max / self.InScale[output_name]
        scale_w = []
        scale_w.append(self.max_range[output_name] / self.s8_max)

        conv_op_node.set_attr("Scale_weights", scale_w)
        conv_op_node.set_attr("Scale_in", scale_in)
        conv_op_node.set_attr("Scale_out", 1.0)
        conv_op_node.set_attr("use_mkldnn", 1)
        conv_op_node.set_attr("force_fp32_output", 1)
        graph.link_to(input_var_node, conv_op_node)
        graph.link_to(weight_var_node, conv_op_node)
        graph.link_to(conv_op_node, output_var_node)
        graph.safe_remove_nodes(op_node)

    def _transform_to_quantize_mkldnn(self, graph, op_node):
        """
        Transform fake_quantize_abs_max op to quantize mkldnn op in the graph.
        """
        input_var_node = graph._find_node_by_name(op_node.inputs,
                                                  op_node.input("X")[0])
        output_var_node = graph._find_node_by_name(op_node.outputs,
                                                   op_node.output("Out")[0])
        if output_var_node.id() in self.mul_input_id:
            return
        else:
            scale_in = self.s8_max / self._load_param(
                self._scope, op_node.input("InScale")[0])[0]
            quant_op_node = graph.create_op_node(
                op_type='quantize',
                attrs={
                    'data_format': 'MKLDNNLAYOUT',
                    'use_mkldnn': 1,
                    'Scale': scale_in,
                    'is_negative_input': 1
                },
                inputs={'Input': input_var_node},
                outputs={'Output': output_var_node})
            graph.link_to(input_var_node, quant_op_node)
            graph.link_to(quant_op_node, output_var_node)
            graph.safe_remove_nodes(op_node)

    def _remove_fake_dequantize_op(self, graph, op_node):
        input_var_node = graph._find_node_by_name(op_node.inputs,
                                                  op_node.input("X")[0])
        if input_var_node.id() in self.mul_output_id:
            return
        else:
            graph.safe_remove_nodes(op_node)

    def _load_param(self, scope, param_name):
        return np.array(scope.find_var(param_name).get_tensor())

    def _restore_var(self, name, array):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array, self._place)

    def _remove_unused_var_nodes(self, graph):
        all_used_vars = set()
        ops = graph.all_op_nodes()
        for op_node in ops:
            for input_node in op_node.inputs:
                all_used_vars.add(input_node)
            for output_node in op_node.outputs:
                all_used_vars.add(output_node)

        all_used_vars = {n.node for n in all_used_vars}
        all_unused_vars = {
            n
            for n in filter(lambda node: node.node not in all_used_vars,
                            graph.all_var_nodes())
        }
        graph.safe_remove_nodes(all_unused_vars)
