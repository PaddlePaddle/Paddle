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
from ..... import compat as cpt
from .... import core
from ....framework import IrGraph
from ....framework import IrNode
from ....framework import Operator
from .... import unique_name

__all__ = [
    'QuantizationTransformPass', 'QuantizationFreezePass', 'ConvertToInt8Pass',
    'TransformForMobilePass', 'OutScaleForTrainingPass',
    'OutScaleForInferencePass', 'AddQuantDequantPass'
]

_fake_quant_op_list = [
    'fake_quantize_abs_max', 'fake_quantize_range_abs_max',
    'fake_quantize_moving_average_abs_max', 'fake_channel_wise_quantize_abs_max'
]

_fake_dequant_op_list = [
    'fake_dequantize_max_abs', 'fake_channel_wise_dequantize_max_abs'
]

_fake_quant_dequant_op_list = [
    'fake_quantize_dequantize_moving_average_abs_max'
]

_out_scale_op_list = [
    "conv2d", "depthwise_conv2d", "mul", "matmul", "relu", "leaky_relu",
    "relu6", "sigmoid", "tanh", "prelu", "swish", "softmax", "batch_norm",
    "elementwise_add", "pool2d", "reshape2", "transpose2"
]

# list op real input and output names, to avoid processing input such as AxisTensor.
_op_real_in_out_name = {
    "conv2d": [["Input", "Filter"], ["Output"]],
    "depthwise_conv2d": [["Input", "Filter"], ["Output"]],
    "mul": [["X", "Y"], ["Out"]],
    "matmul": [["X", "Y"], ["Out"]],
    "pool2d": [["X"], ["Out"]],
    "elementwise_add": [["X", "Y"], ["Out"]],
    "concat": [["X"], ["Out"]],
    "softmax": [["X"], ["Out"]],
    "argmax": [["X"], ["Out"]],
    "transpose": [["X"], ["Out"]],
    "equal": [["X", "Y"], ["Out"]],
    "gather": [["X"], ["Out"]],
    "greater_equal": [["X", "Y"], ["Out"]],
    "greater_than": [["X", "Y"], ["Out"]],
    "less_equal": [["X", "Y"], ["Out"]],
    "less_than": [["X", "Y"], ["Out"]],
    "mean": [["X"], ["Out"]],
    "not_equal": [["X", "Y"], ["Out"]],
    "reshape": [["X"], ["Out"]],
    "reshape2": [["X"], ["Out"]],
    "transpose2": [["X"], ["Out"]],
    "bilinear_interp": [["X"], ["Out"]],
    "nearest_interp": [["X"], ["Out"]],
    "trilinear_interp": [["X"], ["Out"]],
    "slice": [["Input"], ["Out"]],
    "squeeze": [["X"], ["Out"]],
    "elementwise_sub": [["X", "Y"], ["Out"]],
    "relu": [["X"], ["Out"]],
    "relu6": [["X"], ["Out"]],
    "leaky_relu": [["X"], ["Out"]],
    "prelu": [["X"], ["Out"]],
    "tanh": [["X"], ["Out"]],
    "swish": [["X"], ["Out"]],
    "dropout": [["X"], ["Out"]],
    "batch_norm": [["X"], ["Y"]],
    "sigmoid": [["X"], ["Y"]],
}


def _get_op_input_var_names(op):
    """ """
    assert isinstance(op, (IrNode, Operator)), \
        "The input op should be IrNode or Operator."
    var_names = []
    op_name = op.name() if isinstance(op, IrNode) \
        else op.type
    name_list = _op_real_in_out_name[op_name][0]
    for name in name_list:
        var_name = op.input(name)
        if isinstance(var_name, list):
            var_names.extend(var_name)
        else:
            var_names.append(var_name)
    return var_names


def _get_op_output_var_names(op):
    """ """
    assert isinstance(op, (IrNode, Operator)), \
        "The input op should be IrNode or Operator."
    var_names = []
    op_name = op.name() if isinstance(op, IrNode) \
        else op.type
    name_list = _op_real_in_out_name[op_name][1]
    for name in name_list:
        var_name = op.output(name)
        if isinstance(var_name, list):
            var_names.extend(var_name)
        else:
            var_names.append(var_name)
    return var_names


def _init_var_node(var_node, value, scope, place):
    assert isinstance(value,
                      np.ndarray), 'The type of value should be numpy array.'
    assert scope is not None, \
        'The scope cannot be set None.'
    assert place is not None, \
        'The place cannot be set None.'
    tensor = scope.var(var_node.name()).get_tensor()
    tensor.set(value, place)


def _is_input_all_not_persistable(graph, op_node):
    '''
    Analyse the real inputs of the op node are all not persistable.
    '''
    is_input_all_not_persistable = True
    for var_name in _get_op_input_var_names(op_node):
        in_node = graph._find_node_by_name(op_node.inputs, var_name)
        is_input_all_not_persistable = (is_input_all_not_persistable and \
            (not in_node.persistable()))
    return is_input_all_not_persistable


class QuantizationTransformPass(object):
    """
    Quantize the ops that have weights. Add quant and dequant ops for the quantized
    ops's inputs.
    """
    _supported_quantizable_op_type = [
        'conv2d', 'depthwise_conv2d', 'mul', 'matmul'
    ]

    def __init__(self,
                 scope=None,
                 place=None,
                 weight_bits=8,
                 activation_bits=8,
                 activation_quantize_type='abs_max',
                 weight_quantize_type='abs_max',
                 window_size=10000,
                 moving_rate=0.9,
                 skip_pattern=['skip_quant'],
                 quantizable_op_type=['conv2d', 'depthwise_conv2d', 'mul']):
        """
        Constructor.

        Args:
            scope(fluid.Scope): When activation use 'range_abs_max' as the quantize
                type, this pass will create some new parameters. The scope is used to
                initialize these new parameters.
            place(fluid.CPUPlace|fluid.CUDAPlace): place is used to initialize new
                parameters described above.
            weight_bits(int): quantization bit number for weights,
                the bias is not quantized.
            activation_bits(int): quantization bit number for activation.
            activation_quantize_type(str): quantization type for activation,
                now support 'abs_max', 'range_abs_max' and 'moving_average_abs_max'.
                If use 'abs_max' mode, the quantization scale will be calculated
                dynamically each step in both training and testing period. If use
                'range_abs_max', a static quantization scale will be calculated
                during training and used in inference.
            weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. The 'range_abs_max'
                usually is not used for weight, since weights are fixed once the
                model is well trained.
            window_size(int): the window size for 'range_abs_max' quantization.
            moving_rate(float): the param for 'moving_average_abs_max' quantization.
            skip_pattern(str or str list): The user-defined quantization skip pattern, which
                will be presented in the name scope of an op. When the skip pattern is
                detected in an op's name scope, the corresponding op will not be quantized. 
            quantizable_op_type(list[str]): List the type of ops that will be quantized. 
                Default is ["conv2d", "depthwise_conv2d", "mul"]. The quantizable_op_type in
                QuantizationFreezePass and ConvertToInt8Pass must be the same as this.

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
        self._skip_pattern = skip_pattern

        quant_type = [
            'abs_max', 'channel_wise_abs_max', 'range_abs_max',
            'moving_average_abs_max'
        ]
        assert activation_quantize_type != 'channel_wise_abs_max', \
            "The activation quantization type does not support 'channel_wise_abs_max'."
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be "
                "'abs_max' or 'range_abs_max' or 'moving_average_abs_max'." %
                (str(activation_quantize_type)))
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be "
                "'abs_max' or 'channel_wise_abs_max' or 'range_abs_max' or 'moving_average_abs_max'."
                % (str(weight_quantize_type)))

        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type
        self._window_size = window_size
        self._moving_rate = moving_rate

        self._quantizable_ops = quantizable_op_type
        for op in self._quantizable_ops:
            assert op in QuantizationTransformPass._supported_quantizable_op_type, \
                op + " is not supported for quantization."
        self._conv_ops = ['conv2d', 'depthwise_conv2d']
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
        Returns:
            None
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        self._is_test = graph.is_test()
        # marked the variable which has been dequantized.
        dequantized_vars = collections.OrderedDict()
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]

        def _quant_preprocess(op_node):
            user_skipped = False
            if isinstance(self._skip_pattern, list):
                user_skipped = op_node.op().has_attr("op_namescope") and \
                               any(pattern in op_node.op().attr("op_namescope") for pattern in self._skip_pattern)
            elif isinstance(self._skip_pattern, str):
                user_skipped = op_node.op().has_attr("op_namescope") and \
                               op_node.op().attr("op_namescope").find(self._skip_pattern) != -1

            if user_skipped:
                op_node.op()._set_attr("skip_quant", True)

        def _transform_forward(graph, op):
            op.op()._set_attr("quantization_type", "qat_with_weight")
            for var_node in op.inputs:
                if var_node.name() not in op.input_arg_names():
                    continue
                if var_node.name() in dequantized_vars:
                    dequant_var_node = dequantized_vars[var_node.name()]
                else:
                    quant_bits = self._weight_bits if var_node.name() in persistable_vars \
                        else self._activation_bits
                    quant_type = self._weight_quantize_type if var_node.name() \
                        in persistable_vars else self._activation_quantize_type
                    if quant_type == 'channel_wise_abs_max':
                        assert var_node.name(
                        ) in persistable_vars, "'channel_wise_abs_max' can only be applied on weights."
                        if op.name() in self._conv_ops:
                            quant_var_node, scale_var_node = self._insert_channel_quant_op(
                                graph, var_node, quant_bits)
                            dequant_var_node = self._insert_channel_dequant_op(
                                graph, quant_var_node, [scale_var_node],
                                [quant_bits])
                        else:
                            quant_var_node, scale_var_node = self._insert_quant_op(
                                graph, var_node, quant_bits, 'abs_max')
                            dequant_var_node = self._insert_dequant_op(
                                graph, quant_var_node, scale_var_node,
                                quant_bits)
                    else:
                        quant_var_node, scale_var_node = self._insert_quant_op(
                            graph, var_node, quant_bits, quant_type)
                        dequant_var_node = self._insert_dequant_op(
                            graph, quant_var_node, scale_var_node, quant_bits)
                    dequantized_vars[var_node.name()] = dequant_var_node
                graph.update_input_link(var_node, dequant_var_node, op)

        def _transform_backward(graph, op):
            for var_node in op.inputs:
                if var_node.name() not in op.input_arg_names():
                    continue
                if var_node.name() in dequantized_vars:
                    dequant_var_node = dequantized_vars[var_node.name()]
                    graph.update_input_link(var_node, dequant_var_node, op)

        if not self._is_test:
            self._create_global_step(graph)
        ops = graph.all_op_nodes()
        # Do the preproccess of quantization, such as skipping some ops
        # for not being quantized.
        for op in ops:
            if op.name() in self._quantizable_ops or \
                    op.name() in self._quantizable_grad_ops:
                _quant_preprocess(op)
        # The process of _transform_forward and _transform_backward is needed in two for loops.
        # The loop for transforming the forward graph:
        for op in ops:
            if op.name() in self._quantizable_ops:
                if not self._is_skip_quant(graph, op):
                    _transform_forward(graph, op)
        # The loop for renaming the inputs of backward op.
        for op in ops:
            if op.name() in self._quantizable_grad_ops:
                _transform_backward(graph, op)
        graph.resolve_hazard()
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
                _init_var_node(
                    global_step_in,
                    np.zeros(
                        [1], dtype='int64'),
                    self._scope,
                    self._place)
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
            shape=[1],
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
        data_type = 'float64' if var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(
            scale_in_node,
            np.array(
                [0.001], dtype=data_type),
            self._scope,
            self._place)

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
            data_type = 'float64' if var_node.dtype(
            ) == core.VarDesc.VarType.FP64 else 'float32'
            _init_var_node(
                scales_node,
                np.zeros(
                    [self._window_size], dtype=data_type),
                self._scope,
                self._place)

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
        data_type = 'float64' if var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(
            scale_in_node,
            np.array(
                [0.001], dtype=data_type),
            self._scope,
            self._place)

        scale_out_node = graph.create_var_node_from_desc(scale_in_node.var())
        ins = {'X': var_node, 'InScale': scale_in_node}
        outs = {'Out': quant_var_node, 'OutScale': scale_out_node}
        if not self._is_test:
            state_in_node = graph.create_persistable_node(
                name=unique_name.generate('state'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            data_type = 'float64' if var_node.dtype(
            ) == core.VarDesc.VarType.FP64 else 'float32'
            _init_var_node(
                state_in_node,
                np.ones(
                    [1], dtype=data_type),
                self._scope,
                self._place)
            accum_in_node = graph.create_persistable_node(
                name=unique_name.generate('accum'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            _init_var_node(
                accum_in_node,
                np.ones(
                    [1], dtype=data_type),
                self._scope,
                self._place)
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

    def _insert_channel_quant_op(self, graph, var_node, quant_bits):
        """
        Insert fake_channel_wise_quantize_abs_max op in the graph.
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
            shape=[var_node.shape()[0]],
            var_dtype=var_node.dtype())
        quant_op_node = graph.create_op_node(
            op_type='fake_channel_wise_quantize_abs_max',
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

    def _insert_channel_dequant_op(self, graph, var_node, scale_var_nodes,
                                   quant_bits):
        """
        Insert fake_channel_wise_dequantize_max_abs in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        dequant_op_node = graph.create_op_node(
            op_type='fake_channel_wise_dequantize_max_abs',
            attrs={
                'quant_bits': quant_bits,
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': var_node,
                    'Scales': scale_var_nodes},
            outputs={'Out': dequant_var_node})
        graph.link_to(var_node, dequant_op_node)
        for scale_n in scale_var_nodes:
            graph.link_to(scale_n, dequant_op_node)
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

    def _is_skip_quant(self, graph, op_node):
        """
        Analyse whether the op node skips quantization.
        """
        is_skip = False
        if op_node.op().has_attr("skip_quant") and \
            op_node.op().attr("skip_quant"):
            is_skip = True
        # if the inputs of mul and matmul are not all persistable, use
        # AddQuantDequantPass to quantize them.
        if op_node.name() in ["mul", "matmul"] and \
            _is_input_all_not_persistable(graph, op_node):
            is_skip = True
        if op_node.op().has_attr("quantization_type") and \
            op_node.op().attr("quantization_type") == "qat_without_weight":
            is_skip = True
        return is_skip


class QuantizationFreezePass(object):
    def __init__(self,
                 scope,
                 place,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max',
                 quantizable_op_type=None):
        """
        The freeze pass is used to adjust the quantize operator order, for example:
            1) `activation -> quant -> dequant -> conv2d` will be frozen into
            `activation -> quant -> conv2d -> dequant`
            2) `weight -> quant -> dequant -> conv2d` will be frozen into `weight -> conv2d`,
            and weight will be scaled offline.

        Args:
            scope(fluid.Scope): scope is used to get the weight tensor values.
            place(fluid.CPUPlace|fluid.CUDAPlace): place is used to restore the weight tensors.
            weight_bits(int): quantization bit number for weights.
            activation_bits(int): quantization bit number for activation.
            weight_quantize_type(str): quantization type for weights, support 'abs_max' and 
                'channel_wise_abs_max'. The 'range_abs_max' usually is not used for weight, 
                since weights are fixed once the model is well trained.
            quantizable_op_type(list[str]): This input param will be removed latter. The pass
                will process all quantized op, so it is not necessary to set the input param.
        """
        assert scope is not None, \
            'The scope cannot be set None.'
        assert place is not None, \
            'The place cannot be set None.'
        self._scope = scope
        self._place = place
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._weight_quantize_type = weight_quantize_type
        self._conv_ops = ['conv2d', 'depthwise_conv2d']
        self._fake_quant_op_names = _fake_quant_op_list
        self._fake_dequant_op_names = _fake_dequant_op_list
        self._op_input_rename_map = collections.OrderedDict()
        self._op_output_rename_map = collections.OrderedDict()
        self._quant_var_scale_map = collections.OrderedDict()

    def apply(self, graph):
        """
        Adjust quantize/dequantize operators order for the inference process.

        Args:
            graph(IrGraph): the applied graph.
        Returns:
            None
        """
        # Get input scales in fake quant op and process weights
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
                    elif self._weight_quantize_type == 'channel_wise_abs_max':
                        param = self._load_var(input_arg_name)
                        if len(param.shape) == 4:  # conv2d or depthwise_conv2d
                            scale_v = []
                            for i in range(param.shape[0]):
                                scale_v.append(np.max(np.abs(param[i])))
                        else:
                            scale_v = np.max(np.abs(param))
                    else:
                        scale_v = self._load_var(
                            op_node.output('OutScale')[0])[0]
                    self._quant_var_scale_map[input_arg_name] = scale_v
                    self._remove_fake_quant_and_dequant_op(graph, op_node)
                    # quantize weight and restore
                    param_v = self._load_var(input_arg_name)
                    quantized_param_v = self._quant(param_v, scale_v,
                                                    self._weight_bits)
                    self._restore_var(input_arg_name, quantized_param_v)
                else:
                    scale_v = graph._find_node_by_name(
                        op_node.outputs, op_node.output('OutScale')[0])
                    self._quant_var_scale_map[input_arg_name] = scale_v

        # Remove all fake dequant op
        ops = graph.all_op_nodes()
        for op_node in ops:
            op_name = op_node.name()
            if op_name in self._fake_dequant_op_names:
                self._remove_fake_quant_and_dequant_op(graph, op_node)

        # Insert post dequant op
        ops = graph.all_op_nodes()
        for op_node in ops:
            op_node_desc = op_node.op()
            if op_node_desc.has_attr("quantization_type") and \
                op_node_desc.attr("quantization_type") == "qat_with_weight":
                if self._weight_quantize_type == 'channel_wise_abs_max' \
                    and op_node.name() in self._conv_ops:
                    self._insert_post_channel_dequant_op(graph, op_node)
                else:
                    self._insert_post_dequant_op(graph, op_node)

        # Rename inputs of the followed ops after inserting dequant_op after fc/conv
        for op_node in ops:
            for var_node in op_node.inputs:
                if var_node.node in self._op_output_rename_map:
                    old_in = var_node
                    new_in = self._op_output_rename_map[var_node.node]
                    graph.update_input_link(old_in, new_in, op_node)

        # remove the unused var node in the graph
        self._remove_unused_var_nodes(graph)
        graph.resolve_hazard()
        return graph

    def _remove_fake_quant_and_dequant_op(self, graph, op_node):
        k = graph._find_node_by_name(op_node.outputs, op_node.output('Out')[0])
        v = graph._find_node_by_name(op_node.inputs, op_node.input('X')[0])
        if v.node not in self._op_input_rename_map:
            self._op_input_rename_map[k.node] = v
        else:
            self._op_input_rename_map[k.node] = self._op_input_rename_map[
                v.node]
        graph.safe_remove_nodes(op_node)

    def _insert_post_channel_dequant_op(self, graph, op_node):
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        for var_node in op_node.inputs:
            name = var_node.name()
            if name not in op_node.input_arg_names():
                continue
            if var_node.node in self._op_input_rename_map:
                old_in = var_node
                new_in = self._op_input_rename_map[var_node.node]
                new_in.clear_outputs()
                graph.update_input_link(old_in, new_in, op_node)
            original_var_name = self._original_var_name(name)
            scale_v = self._quant_var_scale_map[original_var_name]
            if original_var_name in persistable_vars:
                assert isinstance(
                    scale_v,
                    list), 'The scale of parameter %s is not a list.' % (
                        original_var_name)
                channel_scale = np.array(scale_v)
            else:
                assert isinstance(scale_v, IrNode)
                scale_var_node = self._quant_var_scale_map[original_var_name]

        if len(op_node.output_arg_names()) != 1:
            raise ValueError("Only support one output, but op %s has"
                             " more than one output." % (op_node.name()))

        output_var_node = graph._find_node_by_name(
            op_node.outputs, op_node.output_arg_names()[0])
        weight_scale_node = graph.create_persistable_node(
            name=unique_name.generate('channel_scale'),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[channel_scale.shape[0]],
            var_dtype=output_var_node.dtype())
        data_type = 'float64' if output_var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(weight_scale_node,
                       channel_scale.astype(data_type), self._scope,
                       self._place)
        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(output_var_node.name()),
            var_type=output_var_node.type(),
            shape=output_var_node.shape(),
            var_dtype=output_var_node.dtype())
        dequant_op_node = graph.create_op_node(
            op_type='fake_channel_wise_dequantize_max_abs',
            attrs={
                'quant_bits': [self._weight_bits, self._activation_bits],
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={
                'X': output_var_node,
                'Scales': [weight_scale_node, scale_var_node]
            },
            outputs={'Out': dequant_var_node})
        graph.link_to(output_var_node, dequant_op_node)
        graph.link_to(scale_var_node, dequant_op_node)
        graph.link_to(weight_scale_node, dequant_op_node)
        graph.link_to(dequant_op_node, dequant_var_node)
        self._op_output_rename_map[output_var_node.node] = dequant_var_node
        return dequant_var_node

    def _insert_post_dequant_op(self, graph, op_node):
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        max_range = 1
        param_range = (1 << (self._weight_bits - 1)) - 1
        act_range = (1 << (self._activation_bits - 1)) - 1
        for var_node in op_node.inputs:
            name = var_node.name()
            if name not in op_node.input_arg_names():
                continue
            if var_node.node in self._op_input_rename_map:
                old_in = var_node
                new_in = self._op_input_rename_map[var_node.node]
                new_in.clear_outputs()
                graph.update_input_link(old_in, new_in, op_node)
            original_var_name = self._original_var_name(name)
            scale_v = self._quant_var_scale_map[original_var_name]
            if original_var_name in persistable_vars:
                assert self._is_float(
                    scale_v), 'The scale of parameter %s is not a float.' % (
                        original_var_name)
                max_range *= param_range / scale_v
            else:
                max_range *= act_range
                assert isinstance(scale_v, IrNode)
                scale_var_node = self._quant_var_scale_map[original_var_name]

        if len(op_node.output_arg_names()) != 1:
            raise ValueError("Only support one output, but op %s has"
                             " more than one output." % (op_node.name()))

        output_var_node = graph._find_node_by_name(
            op_node.outputs, op_node.output_arg_names()[0])
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
        self._op_output_rename_map[output_var_node.node] = dequant_var_node
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
        if isinstance(scale, list):
            for i, s in enumerate(scale):
                x[i] = np.round(x[i] / s * ((1 << (num_bits - 1)) - 1))
            return x
        else:
            return np.round(x / scale * ((1 << (num_bits - 1)) - 1))


class ConvertToInt8Pass(object):
    def __init__(self, scope, place, quantizable_op_type=None):
        """
        Convert the weights into int8_t type.

        Args:
            scope(fluid.Scope): scope is used to get the weight tensor values.
            place(fluid.CPUPlace|fluid.CUDAPlace): place is used to restore the
                8bits weight tensors.
            quantizable_op_type(list[str]): This input param will be removed latter. The pass
                will process all quantized op, so it is not necessary to set the input param.
        """
        assert scope is not None, \
            'The scope cannot be set None.'
        assert place is not None, \
            'The place cannot be set None.'
        self._scope = scope
        self._place = place

    def apply(self, graph):
        """
        Convert weights' type of the graph. After that, the data type of the
        graph weights is int8_t.

        Args:
            graph(IrGraph): the applied graph.
        Returns:
            None
        """
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        ops = graph.all_op_nodes()
        input_map = {}
        for op_node in ops:
            if op_node.op().has_attr("quantization_type") and \
                op_node.op().attr("quantization_type") == "qat_with_weight":
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
        graph.resolve_hazard()
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
    def __init__(self):
        """
        This pass is used to convert the frozen graph for paddle-mobile execution.
        """
        self._fake_quant_op_names = _fake_quant_op_list
        self._fake_dequant_op_names = _fake_dequant_op_list

    def apply(self, graph):
        """
        Because paddle-mobile use `quantize` an `dequantize` as the names of
        quantize operator and dequantize operator, the `apply` function just
        realize this logic.

        Args:
            graph(IrGraph): the graph will be transformed.
        Returns:
            None
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
        graph.resolve_hazard()
        return graph


class OutScaleForTrainingPass(object):
    def __init__(self, scope=None, place=None, moving_rate=0.9):
        """
        This pass is used for calculating output scales of some operators.
        These output scales may be used by tensorRT or some other inference engines.

        Args:
            scope(fluid.Scope): The scope is used to initialize these new parameters.
            place(fluid.CPUPlace|fluid.CUDAPlace): The place is used to initialize new parameters.
            moving_rate(float): The decay coefficient of moving average. The default value is 0.9.
        """
        self._scope = scope
        self._place = place
        self._moving_rate = moving_rate
        self._is_test = None
        self._teller_set = _out_scale_op_list

    def apply(self, graph):
        """
        Insert the `moving_average_abs_max_scale` op in order to calculate output scales
        of operators in the teller_set.

        Args:
            graph(IrGraph): the target graph.
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        self._is_test = graph.is_test()
        ops = graph.all_op_nodes()
        for op_node in ops:
            name = op_node.name()
            if name in self._teller_set:
                if len(op_node.output_arg_names()) != 1:
                    continue
                in_node = graph._find_node_by_name(
                    op_node.outputs, op_node.output_arg_names()[0])
                out_node = graph.create_var_node_from_desc(in_node.var())
                scale_node = graph.create_persistable_node(
                    name=self._scale_name(in_node.name()),
                    var_type=core.VarDesc.VarType.LOD_TENSOR,
                    shape=[1],
                    var_dtype=in_node.dtype())
                data_type = 'float64' if in_node.dtype() \
                    == core.VarDesc.VarType.FP64 else 'float32'
                _init_var_node(
                    scale_node,
                    np.ones(
                        [1], dtype=data_type),
                    self._scope,
                    self._place)
                ins = {'X': in_node}
                outs = {'Out': out_node, 'OutScale': scale_node}
                if not self._is_test:
                    state_in_node = graph.create_persistable_node(
                        name=unique_name.generate('scale_state@'),
                        var_type=core.VarDesc.VarType.LOD_TENSOR,
                        var_dtype=in_node.dtype(),
                        shape=[1])
                    _init_var_node(
                        state_in_node,
                        np.ones(
                            [1], dtype=data_type),
                        self._scope,
                        self._place)
                    accum_in_node = graph.create_persistable_node(
                        name=unique_name.generate('scale_accum@'),
                        var_type=core.VarDesc.VarType.LOD_TENSOR,
                        var_dtype=in_node.dtype(),
                        shape=[1])
                    _init_var_node(
                        accum_in_node,
                        np.ones(
                            [1], dtype=data_type),
                        self._scope,
                        self._place)
                    state_out_node = graph.create_var_node_from_desc(
                        state_in_node.var())
                    accum_out_node = graph.create_var_node_from_desc(
                        accum_in_node.var())

                    ins['InState'] = state_in_node
                    ins['InAccum'] = accum_in_node
                    outs['OutState'] = state_out_node
                    outs['OutAccum'] = accum_out_node

                attrs = {
                    'moving_rate': self._moving_rate,
                    'is_test': self._is_test,
                    'op_role': core.op_proto_and_checker_maker.OpRole.Forward
                }
                scale_op_node = graph.create_op_node(
                    op_type='moving_average_abs_max_scale',
                    attrs=attrs,
                    inputs=ins,
                    outputs=outs)
                graph.link_to(in_node, scale_op_node)
                graph.link_to(scale_op_node, out_node)
                graph.link_to(scale_op_node, scale_node)
                if not self._is_test:
                    graph.link_to(state_in_node, scale_op_node)
                    graph.link_to(accum_in_node, scale_op_node)
                    graph.link_to(scale_op_node, state_out_node)
                    graph.link_to(scale_op_node, accum_out_node)
        graph.resolve_hazard()
        return graph

    def _scale_name(self, var_name):
        """
        Return the scale name for the var named `var_name`.
        """
        return "%s@scale" % (var_name)


class OutScaleForInferencePass(object):
    def __init__(self, scope=None):
        """
        This pass is used for setting output scales of some operators.
        These output scales may be used by tensorRT or some other inference engines.

        Args:
            scope(fluid.Scope): The scope is used to initialize these new parameters.
        """
        self._scope = scope
        self._teller_set = _out_scale_op_list

    def apply(self, graph):
        """
        Get output scales from the scope and set these scales in op_descs
        of operators in the teller_set.

        Args:
            graph(IrGraph): the target graph.
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        ops = graph.all_op_nodes()
        for op_node in ops:
            name = op_node.name()
            if name in self._teller_set:
                if len(op_node.output_arg_names()) != 1:
                    continue
                scale_name = self._scale_name(op_node.output_arg_names()[0])
                scale_v = np.array(
                    self._scope.find_var(scale_name).get_tensor())[0]
                op_node.op()._set_attr("out_threshold", float(scale_v))
        graph.resolve_hazard()
        return graph

    def _scale_name(self, var_name):
        """
        Return the scale name for the var named `var_name`.
        """
        return "%s@scale" % (var_name)


class AddQuantDequantPass(object):
    """
    Quantize the ops that do not have weights, and add quant_dequant op for the 
    quantized ops's inputs.
    """
    _supported_quantizable_op_type = [
        "pool2d", "elementwise_add", "concat", "softmax", "argmax", "transpose",
        "equal", "gather", "greater_equal", "greater_than", "less_equal",
        "less_than", "mean", "not_equal", "reshape", "reshape2",
        "bilinear_interp", "nearest_interp", "trilinear_interp", "slice",
        "squeeze", "elementwise_sub", "mul", "matmul", "relu", "relu6",
        "leaky_relu", "tanh", "swish"
    ]

    # To be compatible with PaddleSlim, not remove _activation_type for now
    _activation_type = ["relu", "relu6", "leaky_relu", "tanh", "swish"]

    def __init__(self,
                 scope=None,
                 place=None,
                 moving_rate=0.9,
                 quant_bits=8,
                 skip_pattern=["skip_quant"],
                 quantizable_op_type=["elementwise_add", "pool2d"],
                 is_full_quantized=False):
        """
        Constructor.

        Args:
            scope(fluid.Scope): The scope is used to initialize these new parameters.
            place(fluid.CPUPlace|fluid.CUDAPlace): place is used to initialize new
                parameters described above.
            moving_rate(float, optional): the param for 'quant_dequant_moving_average_abs_max' 
                quantization. Default is 0.9.
            quant_bits(int, optional): quantization bit number for activation. Default is 8.
            skip_pattern(str, optional): The user-defined quantization skip pattern, which
                will be presented in the name scope of an op. When the skip pattern is
                detected in an op's name scope, the corresponding op will not be quantized.
                Default is 'skip_quant'.
            quantizable_op_type(list[str], optional): List the type of ops that will be 
                quantized. Default is ["elementwise_add", "pool2d"]. 
            is_full_quantized(bool, optional): If set is_full_quantized as True, apply 
                quantization to all supported quantizable op type. If set is_full_quantized
                as False, only apply quantization to the op type according to the input 
                quantizable_op_type.
        """
        self._scope = scope
        self._place = place
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits
        self._is_test = None
        self._skip_pattern = skip_pattern

        if is_full_quantized:
            self._quantizable_op_type = \
                AddQuantDequantPass._supported_quantizable_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in quantizable_op_type:
                assert op_type in AddQuantDequantPass._supported_quantizable_op_type, \
                    op_type + " is not supported for quantization."
        self._quantizable_grad_op_type = [
            '%s_grad' % (op) for op in self._quantizable_op_type
        ]

        assert self._scope != None, "scope must not be None."
        assert self._place != None, "place must not be None."

    def apply(self, graph):
        """
        Add quant_dequant before some ops, such as the 'elementwise_add' and
        'pool2d' op.

        Args:
            graph(IrGraph): the target graph.
        Returns:
            None
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        self._is_test = graph.is_test()
        dequantized_vars_map = collections.OrderedDict()

        # Forward stage, insert quant_dequant op
        all_op_nodes = graph.all_op_nodes()
        for op_node in all_op_nodes:
            if op_node.name() in self._quantizable_op_type:
                is_skip = False
                if isinstance(self._skip_pattern, list):
                    is_skip = op_node.op().has_attr("op_namescope") and \
                                   any(pattern in op_node.op().attr("op_namescope") for pattern in self._skip_pattern)
                elif isinstance(self._skip_pattern, str):
                    is_skip = op_node.op().has_attr("op_namescope") and \
                                   op_node.op().attr("op_namescope").find(self._skip_pattern) != -1
                is_quantized = op_node.op().has_attr("quantization_type") and \
                    op_node.op().attr("quantization_type") == "qat_with_weight"
                if is_skip or is_quantized or \
                    (not _is_input_all_not_persistable(graph, op_node)):
                    continue

                op_node.op()._set_attr("quantization_type",
                                       "qat_without_weight")
                op_node.op()._set_attr("activation_bits", self._quant_bits)
                arg_names = _get_op_input_var_names(op_node)
                for arg_name in arg_names:
                    in_node = graph._find_node_by_name(op_node.inputs, arg_name)
                    if arg_name in dequantized_vars_map:
                        quant_var_node = dequantized_vars_map[arg_name]
                    else:
                        quant_var_node, _ = \
                            self._inser_quant_dequant_moving_average_abs_max_op(
                            graph, in_node, self._quant_bits)
                        dequantized_vars_map[arg_name] = quant_var_node
                    graph.update_input_link(in_node, quant_var_node, op_node)

        # Backward stage, update input link
        for op_node in all_op_nodes:
            if op_node.name() in self._quantizable_grad_op_type:
                for input_name in op_node.input_arg_names():
                    if input_name in dequantized_vars_map:
                        in_node = graph._find_node_by_name(op_node.inputs,
                                                           input_name)
                        dequant_var_node = dequantized_vars_map[input_name]
                        graph.update_input_link(in_node, dequant_var_node,
                                                op_node)

        graph.resolve_hazard()
        return graph

    def _inser_quant_dequant_moving_average_abs_max_op(self, graph, var_node,
                                                       quant_bits):
        """Insert fake_quantize_dequantize_moving_average_abs_max op.
        """
        quant_var_node = graph.create_var_node(
            name="{}.quant_dequant".format(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        scale_in_node = graph.create_persistable_node(
            name="{}.quant_dequant.scale".format(var_node.name()),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[1],
            var_dtype=var_node.dtype())
        data_type = 'float64' if var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(
            scale_in_node,
            np.array(
                [0.001], dtype=data_type),
            self._scope,
            self._place)

        scale_out_node = graph.create_var_node_from_desc(scale_in_node.var())
        ins = {'X': var_node, 'InScale': scale_in_node}
        outs = {'Out': quant_var_node, 'OutScale': scale_out_node}
        if not self._is_test:
            state_in_node = graph.create_persistable_node(
                name=unique_name.generate('quant_dequant.state'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            data_type = 'float64' if var_node.dtype(
            ) == core.VarDesc.VarType.FP64 else 'float32'
            _init_var_node(
                state_in_node,
                np.ones(
                    [1], dtype=data_type),
                self._scope,
                self._place)
            accum_in_node = graph.create_persistable_node(
                name=unique_name.generate('quant_dequant.accum'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            _init_var_node(
                accum_in_node,
                np.ones(
                    [1], dtype=data_type),
                self._scope,
                self._place)
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
            op_type='fake_quantize_dequantize_moving_average_abs_max',
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
