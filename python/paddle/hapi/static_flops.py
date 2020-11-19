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

import copy
import numpy as np
import paddle
from collections import OrderedDict
from paddle.static import Program, program_guard, Variable

OPTIMIZER_OPS = [
    'momentum',
    'lars_momentum',
    'adagrad',
    'adam',
    'adamax',
    'dpsgd',
    'decayed_adagrad',
    'adadelta',
    'rmsprop',
]


class VarWrapper(object):
    def __init__(self, var, graph):
        assert isinstance(var, Variable)
        assert isinstance(graph, GraphWrapper)
        self._var = var
        self._graph = graph

    def __eq__(self, v):
        """
        Overwrite this function for ...in... syntax in python.
        """
        return (v is not None) and self._var.name == v._var.name

    def name(self):
        """
        Get the name of the variable.
        """
        return self._var.name

    def __repr__(self):
        return self._var.name

    def shape(self):
        """
        Get the shape of the varibale.
        """
        return self._var.shape

    def set_shape(self, shape):
        """
        Set the shape of the variable.
        """
        self._var.desc.set_shape(shape)

    def inputs(self):
        """
        Get all the operators that use this variable as output.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for op in self._graph.ops():
            if self in op.all_outputs():
                ops.append(op)
        return ops

    def outputs(self):
        """
        Get all the operators that use this variable as input.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for op in self._graph.ops():
            if self in op.all_inputs():
                ops.append(op)
        return ops


class OpWrapper(object):
    def __init__(self, op, graph):
        assert isinstance(graph, GraphWrapper)
        self._op = op
        self._graph = graph

    def __eq__(self, op):
        """
        Overwrite this function for ...in... syntax in python.
        """
        return self.idx() == op.idx()

    def all_inputs(self):
        """
        Get all the input variables of this operator.
        """
        return [
            self._graph.var(var_name) for var_name in self._op.input_arg_names
        ]

    def all_outputs(self):
        """
        Get all the output variables of this operator.
        """
        return [
            self._graph.var(var_name) for var_name in self._op.output_arg_names
        ]

    def idx(self):
        """
        Get the id of this operator.
        """
        return self._op.idx

    def type(self):
        """
        Get the type of this operator.
        """
        return self._op.type

    def __repr__(self):
        return "op[id: {}, type: {}; inputs: {}]".format(self.idx(),
                                                         self.type(),
                                                         self.all_inputs())

    def is_bwd_op(self):
        """
        Whether this operator is backward op.
        """
        return self.type().endswith('_grad')

    def is_opt_op(self):
        """
        Whether this operator is optimizer op.
        """
        return self.type() in OPTIMIZER_OPS

    def inputs(self, name):
        """
        Get all the varibales by the input name.
        """
        if name in self._op.input_names:
            return [
                self._graph.var(var_name) for var_name in self._op.input(name)
            ]
        else:
            return []

    def outputs(self, name):
        """
        Get all the varibales by the output name.
        """
        return [self._graph.var(var_name) for var_name in self._op.output(name)]

    def set_attr(self, key, value):
        """
        Set the value of attribute by attribute's name.

        Args:
            key(str): the attribute name.
            value(bool|int|str|float|list): the value of the attribute.
        """
        self._op._set_attr(key, value)

    def attr(self, name):
        """
        Get the attribute by name.

        Args:
            name(str): the attribute name.

        Returns:
            bool|int|str|float|list: The attribute value. The return value
            can be any valid attribute type.
        """
        return self._op.attr(name)


class GraphWrapper(object):
    """
    It is a wrapper of paddle.fluid.framework.IrGraph with some special functions
    for paddle slim framework.

    Args:
        program(framework.Program): A program with 
        in_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
        out_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
    """

    def __init__(self, program=None, in_nodes=[], out_nodes=[]):
        """
        """
        super(GraphWrapper, self).__init__()
        self.program = Program() if program is None else program
        self.persistables = {}
        self.teacher_persistables = {}
        for var in self.program.list_vars():
            if var.persistable:
                self.persistables[var.name] = var
        self.compiled_graph = None
        in_nodes = [] if in_nodes is None else in_nodes
        out_nodes = [] if out_nodes is None else out_nodes
        self.in_nodes = OrderedDict(in_nodes)
        self.out_nodes = OrderedDict(out_nodes)
        self._attrs = OrderedDict()

    def all_parameters(self):
        """
        Get all the parameters in this graph.

        Returns:
            list<VarWrapper>: A list of VarWrapper instances.
        """
        params = []
        for block in self.program.blocks:
            for param in block.all_parameters():
                params.append(VarWrapper(param, self))
        return params

    def is_persistable(self, var):
        """
        Whether the given variable is persistable.

        Args:
            var(VarWrapper): The given varibale.
        """
        return var._var.persistable

    def ops(self):
        """
        Return all operator nodes included in the graph as a set.
        """
        ops = []
        for block in self.program.blocks:
            for op in block.ops:
                ops.append(OpWrapper(op, self))
        return ops

    def vars(self):
        """
        Get all the variables.
        """
        return [VarWrapper(var, self) for var in self.program.list_vars()]

    def var(self, name):
        """
        Get the variable by variable name.
        """
        for block in self.program.blocks:
            if block.has_var(name):
                return VarWrapper(block.var(name), self)
        return None

    def clone(self, for_test=False):
        """
        Clone a new graph from current graph.

        Returns:
            (GraphWrapper): The wrapper of a new graph.
        """
        return GraphWrapper(
            self.program.clone(for_test),
            copy.deepcopy(self.in_nodes), copy.deepcopy(self.out_nodes))

    def program(self):
        """
        Get the program in current wrapper.
        """
        return self.program

    def pre_ops(self, op):
        """
        Get all the previous operators of target operator.

        Args:
            op(OpWrapper): Target operator.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for p in self.ops():
            for in_var in op.all_inputs():
                if in_var in p.all_outputs():
                    ops.append(p)
        return ops

    def next_ops(self, op):
        """
        Get all the next operators of target operator.

        Args:
            op(OpWrapper): Target operator.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for p in self.ops():
            for out_var in op.all_outputs():
                if out_var in p.all_inputs():
                    ops.append(p)
        return ops


def count_convNd(op):
    filter_shape = op.inputs("Filter")[0].shape()
    filter_ops = np.product(filter_shape[1:])
    bias_ops = 1 if len(op.inputs("Bias")) > 0 else 0
    output_numel = np.product(op.outputs("Output")[0].shape()[1:])
    total_ops = output_numel * (filter_ops + bias_ops)
    return total_ops


def count_leaky_relu(op):
    total_ops = np.product(op.outputs("Output")[0].shape()[1:])
    return total_ops


def count_bn(op):
    output_numel = np.product(op.outputs("Y")[0].shape()[1:])
    total_ops = 2 * output_numel
    return total_ops


def count_linear(op):
    total_mul = op.inputs("Y")[0].shape()[0]
    numel = np.product(op.outputs("Out")[0].shape()[1:])
    total_ops = total_mul * numel
    return total_ops


def count_pool2d(op):
    input_shape = op.inputs("X")[0].shape()
    output_shape = op.outputs('Out')[0].shape()
    kernel = np.array(input_shape[2:]) // np.array(output_shape[2:])
    total_add = np.product(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = np.product(output_shape[1:])
    total_ops = kernel_ops * num_elements
    return total_ops


def count_element_op(op):
    input_shape = op.inputs("X")[0].shape()
    total_ops = np.product(input_shape[1:])
    return total_ops


def _graph_flops(graph, detail=False):
    assert isinstance(graph, GraphWrapper)
    flops = 0
    params2flops = {}
    for op in graph.ops():
        if op.type() in ['conv2d', 'depthwise_conv2d']:
            op_flops = count_convNd(op)
            flops += op_flops
            params2flops[op.inputs("Filter")[0].name()] = op_flops
        elif op.type() == 'pool2d':
            op_flops = count_pool2d(op)
            flops += op_flops

        elif op.type() in ['mul', 'matmul']:
            op_flops = count_linear(op)
            flops += op_flops
            params2flops[op.inputs("Y")[0].name()] = op_flops
        elif op.type() == 'batch_norm':
            op_flops = count_bn(op)
            flops += op_flops
        elif op.type().startswith('element'):
            op_flops = count_element_op(op)
            flops += op_flops
        if op_flops != 0:
            print(op.type())
            print(op_flops)
        op_flops = 0
    if detail:
        return flops, params2flops
    else:
        return flops


def static_flops(program, detail=False):
    graph = GraphWrapper(program)
    return _graph_flops(graph, detail=detail)
