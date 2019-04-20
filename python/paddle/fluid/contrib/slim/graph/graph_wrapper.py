# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict
from .... import io
from .... import compiler
from ....framework import Program
from ....framework import program_guard
from ....framework import Parameter
from ....framework import Variable
from ....executor import Executor
import copy
from collections import Iterable
from ....io import save_inference_model, load_inference_model, save_persistables
import numpy as np
import pickle
import os

__all__ = ['GraphWrapper', 'VarWrapper', 'OpWrapper']

OPTIMIZER_OPS = [
    'momentum',
    'lars_momentum',
    'adagrad',
    'adam',
    'adamax',
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
        return self._var.name == v._var.name

    def name(self):
        """
        Get the name of the variable.
        """
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
            if self in op.all_inputs():
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
            if self in op.all_outputs():
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
        return [self._graph.var(var_name) for var_name in self._op.input(name)]

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
    """

    def __init__(self, program=None, in_nodes=[], out_nodes=[]):
        """
        Args:
            program(framework.Program): A program with 
            in_nodes(dict): A dict to indicate the input nodes of the graph.
                            The key is user-defined and human-readable name.
                            The value is the name of Variable.
            out_nodes(dict): A dict to indicate the input nodes of the graph.
                            The key is user-defined and human-readable name.
                            The value is the name of Variable.
        """
        super(GraphWrapper, self).__init__()
        self.program = Program() if program is None else program
        self.persistables = {}
        for var in self.program.list_vars():
            if var.persistable:
                self.persistables[var.name] = var
        self.compiled_graph = None
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

    def is_parameter(self, var):
        """
        Whether the given variable is parameter.
        Args:
            var(VarWrapper): The given varibale.
        """
        return isinstance(var._var, Parameter)

    def is_persistable(self, var):
        """
        Whether the given variable is persistable.
        Args:
            var(VarWrapper): The given varibale.
        """
        return var._var.persistable

    def compile(self, for_parallel=True, for_test=False):
        """
        Compile the program in this wrapper to framework.CompiledProgram for next running.
        This function must be called if the program is modified.
        Args:
            for_parallel(bool): Whether the program to run in data parallel way. default: True.
            for_test(bool): Whether the compiled program is used for test.
        """
        target = self.program
        if for_test:
            loss = None
        else:
            loss = self.out_nodes['loss']
        if for_parallel:
            # disable memory optimize for stable training
            build_strategy = compiler.BuildStrategy()
            build_strategy.enable_inplace = False
            build_strategy.memory_optimize = False
            build_strategy.relu_memory_optimize = False
            self.compiled_graph = compiler.CompiledProgram(
                target).with_data_parallel(
                    loss_name=loss, build_strategy=build_strategy)
        else:
            self.compiled_graph = compiler.CompiledProgram(target)

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
        return VarWrapper(self.program.global_block().var(name), self)

    def clone(self, for_test=False):
        """
        Clone a new graph from current graph.
        Returns:
            (GraphWrapper): The wrapper of a new graph.
        """
        return GraphWrapper(
            self.program.clone(for_test),
            copy.deepcopy(self.in_nodes), copy.deepcopy(self.out_nodes))

    def merge(self, graph):
        """
        Merge a graph into current graph.
        Args:
            graph(GraphWrapper): The graph to be merged by current graph.
        """
        for var in graph.program.list_vars():
            new_var = self.program.global_block()._clone_variable(
                var, force_persistable=False)
            new_var.stop_gradient = var.stop_gradient
            # TODO: parameters should be cloned
        for op in graph.ops():
            op = op._op
            inputs = {}
            outputs = {}
            attrs = {}
            for input_name in op.input_names:
                inputs[input_name] = [
                    self.var(in_var_name)._var
                    for in_var_name in op.input(input_name)
                ]
            for output_name in op.output_names:
                outputs[output_name] = [
                    self.var(out_var_name)._var
                    for out_var_name in op.output(output_name)
                ]
            for attr_name in op.attr_names:
                attrs[attr_name] = op.attr(attr_name)
            self.program.global_block().append_op(
                type=op.type, inputs=inputs, outputs=outputs, attrs=attrs)

    def program(self):
        """
        Get the program in current wrapper.
        """
        return self.program

    def pre_ops(self, op):
        """
        Get all the previous operators of target operator.
        Args:
            op(OpWrapper): Target operator..
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
            op(OpWrapper): Target operator..
        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for p in self.ops():
            for out_var in op.all_outputs():
                if out_var in p.all_inputs():
                    ops.append(p)
        return ops

    def get_param_by_op(self, op):
        """
        Get the parameters used by target operator.
        """
        assert isinstance(op, OpWrapper)
        params = []
        for var in op.all_inputs():
            if isinstance(var._var, Parameter):
                params.append(var)
        assert len(params) > 0
        return params

    def numel_params(self):
        """
        Get the number of elements in all parameters.
        """
        ret = 0
        for param in self.all_parameters():
            ret += np.product(param.shape())
        return ret

    def get_optimize_graph(self, optimizer, place, scope, no_grad_var_names=[]):
        """
        Get a new graph for training by appending some backward operators and optimization operators.
        Args:
            optimizer: The optimzier used to generate training graph.
            place: The place to run the graph.
            scope: The scope used to run the graph. Some new variable will be added into this scope.
            no_grad_var_names(list<str>): Names of variables that should be ignored while computing gradients. default: [].
        Returns:
            (GraphWrapper): The wrapper of new graph with backward ops and optimization ops. 
        """
        graph = self.clone()
        startup_program = Program()
        with program_guard(
                main_program=graph.program, startup_program=startup_program):
            target_name = None
            if 'loss' in graph.out_nodes:
                target_name = graph.out_nodes['loss']
            elif 'cost' in graph.out_nodes:
                target_name = graph.out_nodes['cost']
            target = graph.var(target_name)._var
            # The learning rate variable may be created in other program.
            # Update information in optimizer to make
            # learning rate variable being accessible in current program.
            if isinstance(optimizer._learning_rate, Variable):
                optimizer._learning_rate_map[
                    graph.program] = optimizer._learning_rate
            optimizer.minimize(target, no_grad_set=no_grad_var_names)

        exe = Executor(place)
        exe.run(program=startup_program, scope=scope)
        return graph

    def flops(self, only_conv=False):
        """
        Get the flops of current graph.
        Args:
            only_conv: Only calculating the conv layers. default: False.
        Returns:
            int: The flops of current graph.
        """
        flops = 0
        for op in self.ops():
            if op.type() in ['conv2d', 'depthwise_conv2d']:
                filter_shape = op.inputs("Filter")[0].shape()
                input_shape = op.inputs("Input")[0].shape()
                output_shape = op.outputs("Output")[0].shape()
                c_out, c_in, k_h, k_w = filter_shape
                _, _, h_out, w_out = output_shape
                groups = op.attr("groups")
                kernel_ops = k_h * k_w * (c_in / groups)
                if len(op.inputs("Bias")) > 0:
                    with_bias = 1
                else:
                    with_bias = 0
                flops += 2 * h_out * w_out * c_out * (kernel_ops + with_bias)
            elif op.type() == 'pool2d' and not only_conv:
                input_shape = op.inputs("X")[0].shape()
                output_shape = op.outputs("Out")[0].shape()
                _, c_out, h_out, w_out = output_shape
                k_size = op.attr("ksize")
                flops += h_out * w_out * c_out * (k_size[0]**2)

            elif op.type() == 'mul' and not only_conv:
                x_shape = list(op.inputs("X")[0].shape())
                y_shape = op.inputs("Y")[0].shape()
                if x_shape[0] == -1:
                    x_shape[0] = 1
                flops += 2 * x_shape[0] * x_shape[1] * y_shape[1]

            elif op.type() in ['relu', 'sigmoid', 'batch_norm'
                               ] and not only_conv:
                input_shape = list(op.inputs("X")[0].shape())
                if input_shape[0] == -1:
                    input_shape[0] = 1
                flops += np.product(input_shape)

        return flops

    def save_persistables(self, path, exe):
        """
        Save all the persistable variables into file.
        Args:
            path(str): The path to save the persistables.
            exe(framework.Executor): The executor used to save the persistables.
        """
        # update persistables from program
        for var in self.program.list_vars():
            if var.persistable and var.name not in self.persistables:
                self.persistables[var.name] = var

        io.save_vars(exe.exe, path, vars=self.persistables.values())

    def load_persistables(self, path, exe):
        """
        Load the persistable variables from file.
        Args:
            path(str): The path to load the persistables.
            exe(framework.Executor): The executor used to load the persistables.
        """

        def if_exist(var):
            return os.path.exists(os.path.join(path, var.name))

        io.load_vars(
            exe.exe, path, vars=self.persistables.values(), predicate=if_exist)

    def update_param_shape(self, scope):
        """
        Update the shape of parameters in the graph according to tensors in scope.
        It is used after loading pruned parameters from file.
        """
        for param in self.all_parameters():
            tensor_shape = np.array(scope.find_var(param.name()).get_tensor(
            )).shape
            param.set_shape(tensor_shape)

    def infer_shape(self):
        """
        Update the groups of convolution layer according to current filters.
        It is used after loading pruned parameters from file.
        """
        for op in self.ops():
            if op.type() != 'conditional_block':
                op._op.desc.infer_shape(op._op.block.desc)

    def update_groups_of_conv(self):
        for op in self.ops():
            if op.type() == 'depthwise_conv2d':
                op.set_attr('groups', op.inputs('Filter')[0].shape()[0])
