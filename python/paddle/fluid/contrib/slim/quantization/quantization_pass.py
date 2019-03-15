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
import six
from ..... import compat as cpt
from .... import core
from .... import Executor
from ....framework import IrGraph
from ....framework import IrNode
from ....framework import Program
from ....initializer import Constant
from .... import unique_name

__all__ = [
    'QuantizationTransformPass', 'QuantizationFreezePass', 'ConvertToInt8Pass',
    'TransformForMobilePass'
]


class QuantizationTransformPass(object):
    def __init__(self,
                 scope=None,
                 place=None,
                 weight_bits=8,
                 activation_bits=8,
                 activation_quantize_type='abs_max',
                 weight_quantize_type='abs_max',
                 window_size=10000,
                 moving_rate=0.9):
        """
        Convert and rewrite the IrGraph according to weight and
        activation quantization type.

        Args:
            scope(fluid.Scope): When activation use 'range_abs_max' as the quantize
            type, this pass will create some new parameters. The scope is used to
            initialize these new parameters.
            place(fluid.CPUPlace|fluid.CUDAPlace): place is used to initialize new
            parameters described above.
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
            # The original graph will be rewrite.
            import paddle.fluid as fluid
            from paddle.fluid.contrib.slim.quantization \
                import QuantizationTransformPass
            from paddle.fluid.contrib.slim.graph import IrGraph
            from paddle.fluid import core

            graph = IrGraph(core.Graph(program.desc), for_test=False)
            place = fluid.CPUPlace()
            transform_pass = QuantizationTransformPass(fluid.global_scope(),
            place)
            transform_pass.apply(graph)
        """
        self._scope = scope
        self._place = place
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits

        quant_type = ['abs_max', 'range_abs_max', 'moving_average_abs_max']
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be ",
                "'abs_max' or 'range_abs_max' or 'moving_average_abs_max'.",
                str(activation_quantize_type))
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be ",
                "'abs_max' or 'range_abs_max' or 'moving_average_abs_max'.",
                str(weight_quantize_type))

        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type
        self._window_size = window_size
        self._moving_rate = moving_rate

        self._need_initialized = collections.OrderedDict()
        self._quantizable_ops = ['conv2d', 'depthwise_conv2d', 'mul']
        self._quantizable_grad_ops = [
            '%s_grad' % (op) for op in self._quantizable_ops
        ]
        self._is_test = None
        self._global_step = None

    def apply(self, graph):
        """
        Quantize the graph for training process. According to weight and
        activation quantization type, the graph will be added some fake
        quantize operators and fake dequantize operators.

        Args:
            graph(IrGraph): the applied graph.
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        self._need_initialized.clear()
        self._is_test = graph.is_test()
        # marked the variable which has been dequantized.
        dequantized_vars = collections.OrderedDict()
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]

        def _transform_forward(graph, op):
            for var_node in op.inputs:
                if var_node.name() in dequantized_vars:
                    dequant_var_node = dequantized_vars[var_node.name()]
                else:
                    quant_bits = self._weight_bits if var_node.name() in persistable_vars \
                    else self._activation_bits
                    quant_type = self._weight_quantize_type if var_node.name() \
                        in persistable_vars else self._activation_quantize_type
                    quant_var_node, scale_var_node = self._insert_quant_op(
                        graph, var_node, quant_bits, quant_type)
                    dequant_var_node = self._insert_dequant_op(
                        graph, quant_var_node, scale_var_node, quant_bits)
                    dequantized_vars[var_node.name()] = dequant_var_node
                graph.update_input_link(var_node, dequant_var_node, op)

        def _transform_backward(graph, op):
            no_dequanted_input_vars = True
            for var_node in op.inputs:
                if var_node.name() in dequantized_vars:
                    dequant_var_node = dequantized_vars[var_node.name()]
                    graph.update_input_link(var_node, dequant_var_node, op)
                    no_dequanted_input_vars = False
            if no_dequanted_input_vars:
                raise ValueError("There is no dequanted inputs for op %s." %
                                 (op.name()))

        if not self._is_test:
            self._create_global_step(graph)
        ops = graph.all_op_nodes()
        # The process of _transform_forward and _transform_backward is needed in two for loops.
        # The loop for transforming the forward graph:
        for op in ops:
            if op.name() in self._quantizable_ops:
                _transform_forward(graph, op)
        # The loop for renaming the inputs of backward op.
        for op in ops:
            if op.name() in self._quantizable_grad_ops:
                _transform_backward(graph, op)

        if len(self._need_initialized) > 0:
            assert self._scope is not None, \
            'The scope cannot be set None when activation_quantize_type equals to range_abs_max.'
            assert self._place is not None, \
            'The place cannot be set None when activation_quantize_type equals to range_abs_max.'
            init_program = Program()
            for var_desc, initializer in six.iteritems(self._need_initialized):
                var = init_program.global_block().create_var(
                    name=var_desc.name(),
                    shape=var_desc.shape(),
                    dtype=var_desc.dtype(),
                    type=var_desc.type(),
                    lod_level=var_desc.lod_level(),
                    persistable=var_desc.persistable())
                initializer(var, init_program.global_block())
            exe = Executor(self._place)
            exe.run(program=init_program, scope=self._scope)

        return graph

    def _create_global_step(self, graph):
        if self._weight_quantize_type == 'range_abs_max' or \
                self._activation_quantize_type == 'range_abs_max':
            counter_name = cpt.to_text('@STEP_COUNTER@')
            for node in graph.all_var_nodes():
                if node.name() == counter_name:
                    self._global_step = node
            if self._global_step is None:
                global_step_in = graph.create_persistable_node(
                    name=counter_name,
                    var_type=core.VarDesc.VarType.LOD_TENSOR,
                    shape=[1],
                    var_dtype=core.VarDesc.VarType.INT64)
                self._need_initialized[global_step_in.var()] = \
                    Constant(value=0, force_cpu=True)
                global_step_out = graph.create_var_node_from_desc(
                    global_step_in.var())
                # The attribute of `op_role` is needed by ParallelExecutor.
                increment_op = graph.create_op_node(
                    op_type='increment',
                    attrs={
                        'step': 1.0,
                        'op_role':
                        core.op_proto_and_checker_maker.OpRole.Forward
                    },
                    inputs={'X': global_step_in},
                    outputs={'Out': global_step_out})
                graph.link_to(global_step_in, increment_op)
                graph.link_to(increment_op, global_step_out)
                self._global_step = global_step_out

    def _insert_quant_op(self, graph, var_node, quant_bits, quant_type):
        """
        Insert fake_quantize_op in the graph.
        """
        if quant_type == 'abs_max':
            return self._insert_quant_abs_max_op(graph, var_node, quant_bits)
        elif quant_type == 'range_abs_max':
            return self._insert_quant_range_abs_max_op(graph, var_node,
                                                       quant_bits)
        elif quant_type == 'moving_average_abs_max':
            return self._insert_quant_moving_average_abs_max_op(graph, var_node,
                                                                quant_bits)

    def _insert_quant_abs_max_op(self, graph, var_node, quant_bits):
        """
        Insert fake_quantize_abs_max op in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        scale_var_node = graph.create_var_node(
            name=self._quantized_scale_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_abs_max',
            attrs={
                'bit_length': quant_bits,
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': var_node},
            outputs={'Out': quant_var_node,
                     'OutScale': scale_var_node})
        graph.link_to(var_node, quant_op_node)
        graph.link_to(quant_op_node, quant_var_node)
        graph.link_to(quant_op_node, scale_var_node)
        return quant_var_node, scale_var_node

    def _insert_quant_range_abs_max_op(self, graph, var_node, quant_bits):
        """
        Insert fake_quantize_range_abs_max on the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())

        scale_in_node = graph.create_persistable_node(
            name=self._quantized_scale_name(var_node.name()),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[1],
            var_dtype=var_node.dtype())
        self._need_initialized[scale_in_node.var()] = Constant(value=0.001)

        scale_out_node = graph.create_var_node_from_desc(scale_in_node.var())
        inputs = {'X': var_node, 'InScale': scale_in_node}
        outputs = {'Out': quant_var_node, 'OutScale': scale_out_node}

        if not self._is_test:
            # The name of scales_var_node maybe 'scales_0', 'scales_1', etc.
            scales_node = graph.create_persistable_node(
                name=unique_name.generate('scales'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                shape=[self._window_size],
                var_dtype=var_node.dtype())
            self._need_initialized[scales_node.var()] = Constant(value=0)
            inputs['Iter'] = self._global_step
            outputs['OutScales'] = scales_node
        attrs = {
            'window_size': self._window_size,
            'bit_length': quant_bits,
            'is_test': self._is_test,
            'op_role': core.op_proto_and_checker_maker.OpRole.Forward
        }
        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_range_abs_max',
            attrs=attrs,
            inputs=inputs,
            outputs=outputs)

        graph.link_to(var_node, quant_op_node)
        graph.link_to(scale_in_node, quant_op_node)
        graph.link_to(quant_op_node, quant_var_node)
        graph.link_to(quant_op_node, scale_out_node)

        if not self._is_test:
            graph.link_to(self._global_step, quant_op_node)
            graph.link_to(quant_op_node, scales_node)

        return quant_var_node, scale_out_node

    def _insert_quant_moving_average_abs_max_op(self, graph, var_node,
                                                quant_bits):
        """Insert fake_quantize_moving_average_abs_max
        """
        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        scale_in_node = graph.create_persistable_node(
            name=self._quantized_scale_name(var_node.name()),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[1],
            var_dtype=var_node.dtype())
        self._need_initialized[scale_in_node.var()] = Constant(value=0.001)

        scale_out_node = graph.create_var_node_from_desc(scale_in_node.var())
        ins = {'X': var_node, 'InScale': scale_in_node}
        outs = {'Out': quant_var_node, 'OutScale': scale_out_node}
        if not self._is_test:
            state_in_node = graph.create_persistable_node(
                name=unique_name.generate('state'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            self._need_initialized[state_in_node.var()] = Constant(value=1)
            accum_in_node = graph.create_persistable_node(
                name=unique_name.generate('accum'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            self._need_initialized[accum_in_node.var()] = Constant(value=1)
            state_out_node = graph.create_var_node_from_desc(state_in_node.var(
            ))
            accum_out_node = graph.create_var_node_from_desc(accum_in_node.var(
            ))

            ins['InState'] = state_in_node
            ins['InAccum'] = accum_in_node
            outs['OutState'] = state_out_node
            outs['OutAccum'] = accum_out_node

        attrs = {
            'bit_length': quant_bits,
            'moving_rate': self._moving_rate,
            'is_test': self._is_test,
            'op_role': core.op_proto_and_checker_maker.OpRole.Forward
        }

        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_moving_average_abs_max',
            attrs=attrs,
            inputs=ins,
            outputs=outs)

        graph.link_to(var_node, quant_op_node)
        graph.link_to(scale_in_node, quant_op_node)
        graph.link_to(quant_op_node, quant_var_node)
        graph.link_to(quant_op_node, scale_out_node)

        if not self._is_test:
            graph.link_to(state_in_node, quant_op_node)
            graph.link_to(accum_in_node, quant_op_node)
            graph.link_to(quant_op_node, state_out_node)
            graph.link_to(quant_op_node, accum_out_node)

        return quant_var_node, scale_out_node

    def _insert_dequant_op(self, graph, var_node, scale_var_node, quant_bits):
        """
        Insert fake_dequantize_op in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        max_range = (1 << (quant_bits - 1)) - 1
        dequant_op_node = graph.create_op_node(
            op_type='fake_dequantize_max_abs',
            attrs={
                'max_range': float(max_range),
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': var_node,
                    'Scale': scale_var_node},
            outputs={'Out': dequant_var_node})
        graph.link_to(var_node, dequant_op_node)
        graph.link_to(scale_var_node, dequant_op_node)
        graph.link_to(dequant_op_node, dequant_var_node)
        return dequant_var_node

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
        Return the scale name of quantized variable for the input `var_name`.
        """
        return "%s.scale" % (var_name)


class QuantizationFreezePass(object):
    """
    The freeze pass is used to adjust the quantize operator order, for example:
        1) `activation -> quant -> dequant -> conv2d` will be freezed into
        `activation -> quant -> conv2d -> dequant`
        2) `weight -> quant -> dequant -> conv2d` will be freezed into `weight -> conv2d`,
        and weight will be sacled offline.

    Args:
        scope(fluid.Scope): scope is used to get the weight tensor values.
        place(fluid.CPUPlace|fluid.CUDAPlace): place is used to restore the weight tensors.
        weight_bits (int): quantization bit number for weights.
        activation_bits (int): quantization bit number for activation.
        weight_quantize_type (str): quantization type for weights, support 'abs_max'.
        The 'range_abs_max' usually is not used for weight, since weights are fixed once the
        model is well trained.
    """

    def __init__(self,
                 scope,
                 place,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max'):
        assert scope is not None, \
            'The scope cannot be set None.'
        assert place is not None, \
            'The place cannot be set None.'
        self._scope = scope
        self._place = place
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._weight_quantize_type = weight_quantize_type
        self._quantizable_ops = ['conv2d', 'depthwise_conv2d', 'mul']
        self._fake_quant_op_names = [
            'fake_quantize_abs_max', 'fake_quantize_range_abs_max',
            'fake_quantize_moving_average_abs_max'
        ]
        self._fake_dequant_op_names = ['fake_dequantize_max_abs']
        self._op_input_rename_map = collections.OrderedDict()
        self._op_output_rename_map = collections.OrderedDict()
        self._var_scale_map = collections.OrderedDict()

    def apply(self, graph):
        """
        Adjust quantize/dequantize operators order for the inference process.

        Args:
            graph(IrGraph): the applied graph.
        """
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        ops = graph.all_op_nodes()
        for op_node in ops:
            op_name = op_node.name()
            if op_name in self._fake_quant_op_names:
                input_arg_name = op_node.input('X')[0]
                if input_arg_name in persistable_vars:
                    if self._weight_quantize_type == 'abs_max':
                        param = self._load_var(input_arg_name)
                        scale_v = np.max(np.abs(param))
                    else:
                        scale_v = self._load_var(
                            op_node.output('OutScale')[0])[0]
                    self._var_scale_map[input_arg_name] = scale_v
                else:
                    scale_v = graph.var_node(op_node.output('OutScale')[0])
                    self._var_scale_map[input_arg_name] = scale_v
                if input_arg_name in persistable_vars:
                    self._remove_fake_quant_and_dequant_op(graph, op_node)
                    # quantize weight and restore
                    param_v = self._load_var(input_arg_name)
                    quantized_param_v = self._quant(param_v, scale_v,
                                                    self._weight_bits)
                    self._restore_var(input_arg_name, quantized_param_v)

        ops = graph.all_op_nodes()
        for op_node in ops:
            op_name = op_node.name()
            if op_name in self._fake_dequant_op_names:
                self._remove_fake_quant_and_dequant_op(graph, op_node)

        ops = graph.all_op_nodes()
        for op_node in ops:
            op_name = op_node.name()
            if op_name in self._quantizable_ops:
                self._insert_post_dequant_op(graph, op_node)

        for op_node in ops:
            # insert dequant_op after fc/conv, need to rename inputs of the followed ops
            for var_node in op_node.inputs:
                name = var_node.name()
                if name in self._op_output_rename_map:
                    old_in = graph.var_node(name)
                    new_in = self._op_output_rename_map[name]
                    graph.update_input_link(old_in, new_in, op_node)

        # remove the unused var node in the graph
        self._remove_unused_var_nodes(graph)
        return graph

    def _remove_fake_quant_and_dequant_op(self, graph, op_node):
        k = op_node.output('Out')[0]
        v = op_node.input('X')[0]
        if v not in self._op_input_rename_map:
            self._op_input_rename_map[k] = v
        else:
            self._op_input_rename_map[k] = self._op_input_rename_map[v]
        graph.safe_remove_nodes(op_node)

    def _insert_post_dequant_op(self, graph, op_node):
        max_range = None
        scale_var_node = None
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        for var_node in op_node.inputs:
            name = var_node.name()
            if name in self._op_input_rename_map:
                old_in = graph.var_node(name)
                new_in = graph.var_node(self._op_input_rename_map[name])
                new_in.clear_outputs()
                graph.update_input_link(old_in, new_in, op_node)
            original_var_name = self._original_var_name(name)
            scale_v = self._var_scale_map[original_var_name]
            if original_var_name in persistable_vars:
                param_range = (1 << (self._weight_bits - 1)) - 1
                act_range = (1 << (self._activation_bits - 1)) - 1
                assert self._is_float(
                    scale_v), 'The scale of parameter %s is not a float.' % (
                        original_var_name)
                max_range = param_range * act_range / scale_v
            else:
                assert isinstance(scale_v, IrNode)
                scale_var_node = self._var_scale_map[original_var_name]

        if len(op_node.outputs) != 1:
            raise ValueError("Only support one output, but op %s has"
                             " more than one output." % (op_node.name()))

        output_var_node = op_node.outputs[0]
        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(output_var_node.name()),
            var_type=output_var_node.type(),
            shape=output_var_node.shape(),
            var_dtype=output_var_node.dtype())
        dequant_op_node = graph.create_op_node(
            op_type='fake_dequantize_max_abs',
            attrs={
                'max_range': float(max_range),
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': output_var_node,
                    'Scale': scale_var_node},
            outputs={'Out': dequant_var_node})
        graph.link_to(output_var_node, dequant_op_node)
        graph.link_to(scale_var_node, dequant_op_node)
        graph.link_to(dequant_op_node, dequant_var_node)
        self._op_output_rename_map[output_var_node.name()] = dequant_var_node
        return dequant_var_node

    def _load_var(self, name):
        return np.array(self._scope.find_var(name).get_tensor())

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

    def _dequantized_var_name(self, var_name):
        """
        Return dequantized variable name for the input `var_name`.
        """
        return "%s.dequantized" % (var_name)

    def _is_float(self, v):
        return isinstance(v, float) or isinstance(v, np.float32) \
            or isinstance(v, np.float64)

    def _quant(self, x, scale, num_bits):
        return np.round(x / scale * ((1 << (num_bits - 1)) - 1))


class ConvertToInt8Pass(object):
    """
    Convert the weights into int8_t type.

    Args:
        scope(fluid.Scope): scope is used to get the weight tensor values.
        place(fluid.CPUPlace|fluid.CUDAPlace): place is used to restore the
        8bits weight tensors.
    """

    def __init__(self, scope, place):
        assert scope is not None, \
            'The scope cannot be set None.'
        assert place is not None, \
            'The place cannot be set None.'
        self._scope = scope
        self._place = place
        self._quantizable_ops = ['conv2d', 'depthwise_conv2d', 'mul']

    def apply(self, graph):
        """
        Convert weights' tpye of the graph. After that, the data type of the
        graph weigths is int8_t.

        Args:
            graph(IrGraph): the applied graph.
        """
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        ops = graph.all_op_nodes()
        input_map = {}
        for op_node in ops:
            op_name = op_node.name()
            if op_name in self._quantizable_ops:
                for var_node in op_node.inputs:
                    name = var_node.name()
                    if name in persistable_vars:
                        if name not in input_map:
                            int8_var_node = self._convert_to_int8(graph,
                                                                  var_node)
                            input_map[name] = int8_var_node
                        graph.update_input_link(var_node, input_map[name],
                                                op_node)

        # remove the unused var node in the graph
        self._remove_unused_var_nodes(graph)
        return graph

    def _convert_to_int8(self, graph, var_node):
        int8_var_node_name = var_node.name() + ".int8"
        int8_var_node = graph.create_persistable_node(
            name=cpt.to_text(int8_var_node_name),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=core.VarDesc.VarType.INT8)
        array = self._load_var(var_node.name())
        self._scope.var(int8_var_node_name)
        self._store_var(int8_var_node_name, array, np.int8)
        return int8_var_node

    def _load_var(self, name):
        return np.array(self._scope.find_var(name).get_tensor())

    def _store_var(self, name, array, dtype):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array.astype(dtype), self._place)

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


class TransformForMobilePass(object):
    """
    This pass is used to convert the freezed graph for paddle-mobile execution.
    """

    def __init__(self):
        self._fake_quant_op_names = [
            'fake_quantize_abs_max', 'fake_quantize_range_abs_max'
        ]
        self._fake_dequant_op_names = ['fake_dequantize_max_abs']

    def apply(self, graph):
        """
        Because paddle-mobile use `quantize` an `dequantize` as the names of
        quantize operator and dequantize operator, the `apply` function just
        realize this logic.

        Args:
            graph(IrGraph): the graph will be transformed.
        """
        ops = graph.all_op_nodes()
        for op_node in ops:
            name = op_node.name()
            if name in self._fake_quant_op_names:
                op_node.set_type('quantize')
                quant_node = graph.create_op_node_from_desc(op_node.op())
                for input_node in op_node.inputs:
                    graph.link_to(input_node, quant_node)
                for output_node in op_node.outputs:
                    graph.link_to(quant_node, output_node)
                graph.safe_remove_nodes(op_node)
            if name in self._fake_dequant_op_names:
                op_node.set_type('dequantize')
                dequant_node = graph.create_op_node_from_desc(op_node.op())
                for input_node in op_node.inputs:
                    graph.link_to(input_node, dequant_node)
                for output_node in op_node.outputs:
                    graph.link_to(dequant_node, output_node)
                graph.safe_remove_nodes(op_node)

        return graph
