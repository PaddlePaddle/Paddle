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
from .... import core
from ....framework import IrGraph
from ....framework import IrNode

__all__ = ['Qat2Int8MkldnnPass']


class Qat2Int8MkldnnPass(object):
    """
    Transform a QAT model IrGraph into MKL-DNN supported INT8 IrGraph.
    The pass consists of the following transformations:
        1. gather scale values from fake quantize/dequantize operators,
        2. extract FP32 inference model graph from the QAT graph, i.e.
            a.  remove fake quantize/dequantize operators,
            b.  dequantize conv2d and mul's weights,
        3. optimize the FP32 graph using standard FP32 optimization fuses
            (e.g. `conv2d`+`bn` -> `conv2d`),
        4. quantize the optimized FP32 graph using standard INT8v2 quantization
            passes (`cpu_quantize_pass`, `cpu_quantize_squash_pass`).
    """

    def __init__(self,
                 _quantized_ops,
                 _scope=None,
                 _place=None,
                 _core=None,
                 _debug=False):
        self._scope = _scope
        self._place = _place
        self._core = _core
        self._debug = _debug
        self._quantize_types = [
            'fake_quantize_moving_average_abs_max',
            'fake_quantize_range_abs_max',
            'fake_quantize_dequantize_moving_average_abs_max'
        ]
        self._fake_quantize_types = [
            'fake_quantize_moving_average_abs_max',
            'fake_quantize_dequantize_moving_average_abs_max'
        ]
        self._fake_dequantize_types = ['fake_dequantize_max_abs']
        self._quantized_ops = _quantized_ops
        self._scale_immutable_ops = [
            'transpose2', 'reshape2', 'pool2d', 'scale'
        ]
        self._conv_ops = ['conv2d', 'depthwise_conv2d']
        self._pool_ops = ['pool2d']
        self._mul_ops = ['mul']
        self._fc_ops = ['fc']
        self._weight_scales = {}
        # Collect the Input and Output sclaes from Fake QAT models
        self._var_quant_scales = {}
        self._max_range = {}
        self._s8_max = 127

    def apply(self, graph):
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'

        graph = self._gather_scales(graph)
        graph = self._remove_fake_ops(graph)
        graph = self._dequantize_weights(graph)
        graph = self._optimize_fp32_graph(graph)
        graph = self._compute_weight_scales(graph)
        graph = self._update_relu_output_scales(graph)
        graph = self._propagate_scales(graph)
        graph = self._set_dummy_fc_out_scales(graph)
        graph = self._quantize_fp32_graph(graph)
        graph = self._remove_unused_var_nodes(graph)
        return graph

    def apply_fp32(self, graph):
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'

        graph = self._gather_scales(graph)
        graph = self._remove_fake_ops(graph)
        graph = self._dequantize_weights(graph)
        graph = self._optimize_fp32_graph(graph)
        graph = self._remove_unused_var_nodes(graph)
        return graph

    def _convert_scale2tensor(self, scale):
        tensor = core.LoDTensor()
        tensor.set(scale, core.CPUPlace())
        return tensor

    def _is_conv_quantized(self):
        return any(op_type in self._quantized_ops for op_type in self._conv_ops)

    def _is_fc_quantized(self):
        return 'fc' in self._quantized_ops

    def _gather_scales(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._quantize_types:
                bit_length = op.op().attr("bit_length")
                assert bit_length == 8, 'Unsupported number quantization bits ({}). Only 8 is supported now.'.format(
                    bit_length)

                input_name = op.input("X")[0]
                scale_name = op.input("InScale")[0]
                # Gather new weights scale after folding batchnorm in convolution
                scale = np.array(1.0 / self._load_param(
                    self._scope, scale_name)[0]).astype(np.float64)
                lod_tensor = self._convert_scale2tensor(scale)
                use_unsigned_int = False
                self._var_quant_scales[input_name] = (use_unsigned_int,
                                                      lod_tensor)
                self._var_quant_scales[scale_name.replace(".scale", "")] = (
                    use_unsigned_int, lod_tensor)

            if op.name() in self._fake_dequantize_types:
                input_name = op.input("X")[0]
                _max_range = op.op().attr("max_range")
                self._weight_scales[input_name] = _max_range
        return graph

    def _propagate_scales(self, graph):
        def _update_scale_op_in_scale(op, input, output):
            unsigned, tensor = self._var_quant_scales[output]
            scale = np.array(tensor) * op.op().attr("scale")
            new_tensor = self._convert_scale2tensor(scale.astype(np.float64))
            self._var_quant_scales[input] = (unsigned, new_tensor)

        def _update_scales(graph):
            waiting_for_scale = set()
            for op in graph.all_op_nodes():
                if op.name() in self._scale_immutable_ops:
                    input_name = op.input("X")[0]
                    output_name = op.output("Out")[0]
                    tensor_names = [input_name, output_name]

                    # Scale is not quantized, so if it doesn't have any scales
                    # to propagate, its tensors won't be added to the waiting list.
                    if all(name not in self._var_quant_scales for name in tensor_names) \
                            and op.name() != 'scale':
                        waiting_for_scale.update(tensor_names)
                        continue

                    if input_name in self._var_quant_scales:
                        self._var_quant_scales[
                            output_name] = self._var_quant_scales[input_name]
                    elif output_name in self._var_quant_scales:
                        if op.name() == 'scale':
                            _update_scale_op_in_scale(op, input_name,
                                                      output_name)
                        else:
                            self._var_quant_scales[
                                input_name] = self._var_quant_scales[
                                    output_name]
            return waiting_for_scale

        waiting_for_scale = _update_scales(graph)
        waiting_for_scale_prev = set()

        while len(waiting_for_scale
                  ) != 0 and waiting_for_scale != waiting_for_scale_prev:
            waiting_for_scale_prev = waiting_for_scale
            waiting_for_scale = _update_scales(graph)

        return graph

    def _set_dummy_fc_out_scales(self, graph):
        '''
        For the output tensors of FC that do not have an assigned scale,
        assign a dummy scale (same scale as input), so that the quantize pass
        won't fail. In the end these scales aren't used, since FCs that
        have an unassigend output scale will have a force_fp32_output attr
        set to True.
        '''
        for op in graph.all_op_nodes():
            if op.name() in self._fc_ops:
                input_name = op.input("Input")[0]
                output_name = op.output("Out")[0]
                if input_name in self._var_quant_scales and \
                    output_name not in self._var_quant_scales:
                    # use input scale as a "dummy" scale
                    self._var_quant_scales[
                        output_name] = self._var_quant_scales[input_name]

        return graph

    def _load_param(self, scope, param_name):
        return np.array(scope.find_var(param_name).get_tensor())

    def _remove_fake_ops(self, graph):
        '''
        When FC isn't quantized:
        Remove fake (de)quantize ops that do not surround mul.
        When FC is quantized:
        Remove all fake (de)quantize ops.
        '''
        is_fc_quantized = self._is_fc_quantized()
        for op in graph.all_op_nodes():
            if op.name() in self._fake_quantize_types:
                op_out = graph._find_node_by_name(op.outputs,
                                                  op.output("Out")[0])
                next_op = op_out.outputs[0]
                if next_op.name() not in self._mul_ops or is_fc_quantized:
                    self._remove_fake_quantize(graph, op)

        for op in graph.all_op_nodes():
            if op.name() in self._fake_dequantize_types:
                op_in = graph._find_node_by_name(op.inputs, op.input("X")[0])
                prev_op = op_in.inputs[0]
                if prev_op.name() not in self._mul_ops or is_fc_quantized:
                    self._remove_fake_dequantize(graph, op)

        return graph

    def _remove_fake_quantize(self, graph, op):
        fake_quant_in = graph._find_node_by_name(op.inputs, op.input("X")[0])
        fake_quant_in_scale = graph._find_node_by_name(op.inputs,
                                                       op.input("InScale")[0])
        fake_quant_out = graph._find_node_by_name(op.outputs,
                                                  op.output("Out")[0])
        fake_quant_out_scale = graph._find_node_by_name(
            op.outputs, op.output("OutScale")[0])

        next_ops = fake_quant_out.outputs
        for next_op in next_ops:
            self._swap_inputs(next_op, fake_quant_out, fake_quant_in)
            graph.link_to(fake_quant_in, next_op)
        graph.safe_remove_nodes(
            {op, fake_quant_in_scale, fake_quant_out, fake_quant_out_scale})

        return graph

    def _remove_fake_dequantize(self, graph, op):
        fake_dequant_in = graph._find_node_by_name(op.inputs, op.input("X")[0])
        fake_dequant_out = graph._find_node_by_name(op.outputs,
                                                    op.output("Out")[0])

        next_ops = fake_dequant_out.outputs
        for next_op in next_ops:
            self._swap_inputs(next_op, fake_dequant_out, fake_dequant_in)
            graph.link_to(fake_dequant_in, next_op)
        graph.safe_remove_nodes({op, fake_dequant_out})

        return graph

    def _swap_inputs(self, op, old_input, new_input):
        for input_name in op.op().input_names():
            if old_input.name() in op.input(input_name):
                op.op().set_input(input_name, [
                    new_input.name() if x == old_input.name() else x
                    for x in op.input(input_name)
                ])

    def _dequantize_weights(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._conv_ops:
                self._dequantize_conv_weights(graph, op)
            elif self._is_fc_quantized() and op.name() in self._mul_ops:
                self._dequantize_mul_weights(graph, op)
        return graph

    def _dequantize_conv_weights(self, graph, op_node):
        weight_name = op_node.input("Filter")[0]
        output_name = op_node.output("Output")[0]
        # Convert int8 range weights to fp32 range weights
        scales = self._weight_scales[output_name]
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(np.multiply(weight, self._s8_max), scales)
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)

    def _dequantize_mul_weights(self, graph, op_node):
        weight_name = op_node.input("Y")[0]
        output_name = op_node.output("Out")[0]
        scales = self._weight_scales[output_name]
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(np.multiply(weight, self._s8_max), scales)
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)

    def _restore_var(self, name, array):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array, self._place)

    def _update_activations(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._conv_ops and not op.op().has_attr(
                    "fuse_activation"):
                activation = ""
                if op.op().has_attr("fuse_relu") and op.op().attr("fuse_relu"):
                    activation = "relu"
                elif op.op().has_attr("fuse_brelu") and op.op().attr(
                        "fuse_brelu"):
                    activation = "relu6"
                    alpha = 6.0
                    if op.op().has_attr("fuse_brelu_threshold"):
                        alpha = op.op().attr("fuse_brelu_threshold")
                    op.set_attr("fuse_alpha", alpha)
                op.set_attr("fuse_activation", activation)
        return graph

    def _remove_ctrl_vars(self, graph):
        remove_ctr_vars = set()
        for node in graph.all_var_nodes():
            if node.is_ctrl_var():
                remove_ctr_vars.add(node)
        graph.safe_remove_nodes(remove_ctr_vars)
        return graph

    def _optimize_fp32_graph(self, graph):
        graph = self._update_activations(graph)
        graph = self._remove_ctrl_vars(graph)
        graph = self._apply_pass(graph, 'mkldnn_placement_pass',
                                 ['mkldnn_enabled_op_types'], [set()])
        if self._is_conv_quantized():
            graph = self._apply_pass(graph, 'depthwise_conv_mkldnn_pass')
            graph = self._apply_pass(graph, 'conv_bn_fuse_pass')
            graph = self._apply_pass(graph, 'conv_eltwiseadd_bn_fuse_pass')
            graph = self._apply_pass(graph, 'conv_bias_mkldnn_fuse_pass')
            graph = self._apply_pass(graph,
                                     'conv_elementwise_add_mkldnn_fuse_pass')
            graph = self._apply_pass(graph, 'conv_relu_mkldnn_fuse_pass')
            graph = self._apply_pass(graph, 'conv_relu6_mkldnn_fuse_pass')
        if self._is_fc_quantized():
            graph = self._apply_pass(graph, 'fc_fuse_pass',
                                     ['use_gpu', 'use_fc_padding'],
                                     [False, False])
            graph = self._apply_pass(graph, 'fc_mkldnn_pass')
        return graph

    def _apply_pass(self, graph, pass_name, attrs=None, attr_values=None):
        ir_pass = core.get_pass(pass_name)
        cpp_graph = graph.graph
        if not cpp_graph.has('__param_scope__'):
            cpp_graph.set_not_owned('__param_scope__', self._scope)
        if attrs:
            assert attr_values and len(attrs) == len(
                attr_values
            ), "Different number of pass attributes and their values."
            for attr, value in zip(attrs, attr_values):
                ir_pass.set(attr, value)
        ir_pass.apply(cpp_graph)
        if self._debug:
            graph.draw('.', 'qat_fp32_{}'.format(pass_name),
                       graph.all_op_nodes())
        self._remove_unused_var_nodes(graph)
        return graph

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
        return graph

    def _compute_weight_scales(self, graph):
        def _compute_var_scales(ops, out_name, w_name, axis):
            for op in graph.all_op_nodes():
                if op.op().type() in ops:
                    weight_var_name = op.input(w_name)[0]
                    weights = np.array(
                        self._load_param(self._scope, weight_var_name))
                    scales = 1.0 / np.amax(
                        np.abs(weights.reshape(weights.shape[0], -1)).astype(
                            np.float64),
                        axis=axis)
                    scales[scales == np.Inf] = 0.0

                    lod_tensor = self._convert_scale2tensor(scales)
                    use_unsigned_int = False
                    self._var_quant_scales[weight_var_name] = (use_unsigned_int,
                                                               lod_tensor)

        _compute_var_scales(self._conv_ops, "Output", "Filter", axis=1)
        _compute_var_scales(self._fc_ops, "Out", "W", axis=0)
        return graph

    def _find_avg_pooling_ids(self, graph):
        ids = []
        for op in graph.all_op_nodes():
            if op.name() in self._pool_ops:
                if op.op().attr("pooling_type") == "avg":
                    ids.append(op.id())
        return set(ids) if len(ids) else set([-1])

    def _update_relu_output_scales(self, graph):
        def _update_scale(graph, ops, op_out_name, predicate):
            '''
            Sets the type of an output scale of a passed op type(s) to 'unsigned int8' if the
            predicate applied on op passes. Typically, the predicate checks if op's
            activation is set to relu.
            '''
            for op in graph.all_op_nodes():
                if op.name() in ops:
                    out_name = op.output(op_out_name)[0]
                    if out_name in self._var_quant_scales and predicate(op.op(
                    )):
                        _, tensor = self._var_quant_scales[out_name]
                        self._var_quant_scales[out_name] = (True, tensor)
            return graph

        if self._is_conv_quantized():
            conv_predicate = lambda op: op.attr("fuse_activation") == 'relu' and \
                op.attr("fuse_residual_connection") == False
            graph = _update_scale(graph, self._conv_ops, "Output",
                                  conv_predicate)

        if self._is_fc_quantized():
            fc_predicate = lambda op: op.attr("activation_type") == 'relu'
            graph = _update_scale(graph, self._fc_ops, "Out", fc_predicate)

        return graph

    def _get_data_layout(self):
        return 'NHWC' if self._is_conv_quantized() else 'NCHW'

    def _quantize_fp32_graph(self, graph):
        ir_pass = self._core.get_pass('cpu_quantize_placement_pass')
        cpp_graph = graph.graph
        ir_pass.set('quantize_enabled_op_types', self._quantized_ops)
        ir_pass.set('quantize_excluded_op_ids',
                    self._find_avg_pooling_ids(graph))
        ir_pass.apply(cpp_graph)
        if self._debug:
            graph.draw('.', 'qat_int8_{}'.format(ir_pass.type()),
                       graph.all_op_nodes())

        graph = self._apply_pass(
            graph, 'cpu_quantize_pass', ['quant_var_scales', 'data_layout'],
            [self._var_quant_scales, self._get_data_layout()])
        graph = self._apply_pass(graph, 'cpu_quantize_squash_pass')
        return graph
