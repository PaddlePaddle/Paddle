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
from collections import OrderedDict
from .... import io
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

__all__ = [
    'Graph',
    'ImitationGraph',
    'IRGraph',
    'save_inference_graph_model',
    'load_inference_graph_model',
    'load_persistables',
    'save_persistables',
    'update_depthwise_conv',
    'update_param_shape',
    'infer_shape',
]


class Graph(object):
    """
    Base class for all graph.
    """

    def __init__(self):
        pass

    def has(self, attr_name):
        """
        Test attr_name whether in the graph or not.
        """
        pass

    def set(self, attr_name, attr):
        """
        Register attr for graph using the given attr_name.
        """
        pass

    def get(self, attr_name):
        """
        Get attr from the graph by attr_name.
        """
        pass

    def all_parameters(self):
        """
        Return all the parameters in current graph.
        """
        pass

    def create_parameter(self, *args, **kwargs):
        """
        Create a parameter in the graph.
        """
        pass

    def all_vars(self):
        """
        Return all the variables in current graph.
        """
        pass

    def vars_map(self):
        """
        Return the variables map, key-value: var_name --> var
        """
        pass

    def all_ops(self):
        """
        Return all the operations in current graph.
        """
        pass

    def index(self, op):
        """
        Return the index of the op in current graph.
        """
        pass

    def var(self, name):
        """
        Get a Variable by the given name.
        """
        pass

    def create_var(self, *args, **kwargs):
        """
        Create a var in the graph.
        """
        pass

    def remove_var(self, name):
        """
        Remove a var from the graph by the given name.
        """
        pass

    def insert_op(self, index, *args, **kwargs):
        """
        Insert an operation before the index op.
        """
        pass

    def prepend_op(self, *args, **kwargs):
        """
        Insert an operation before the first op.
        """
        pass

    def remove_op(self, index):
        """
        Remove the index operation.
        """
        pass

    def clone(self, for_test=False):
        """
        Create a new duplicated graph.

        Some operators, e.g., :code:`batch_norm`, behave differently between
        training and testing. They have an attribute, :code:`is_test`, to
        control this behaviour. This method will change the :code:`is_test`
        attribute of them to :code:`True` when :code:`for_test=True`.
        """
        pass

    def prune(self, feeds, fetches):
        """
        Prune the graph according to feeds and fetches.
        """
        pass


class ImitationGraph(Graph):
    def __init__(self, program=None, scope=None, in_nodes=[], out_nodes=[]):
        super(ImitationGraph, self).__init__()
        self.program = Program() if program is None else program
        self.compiled_graph = None
        self.scope = scope
        self.in_nodes = OrderedDict(in_nodes)
        self.out_nodes = OrderedDict(out_nodes)
        self._attrs = collections.OrderedDict()

    def init_vars(self, need_inited, place):
        init_program = Program()
        for var, initializer in need_inited.iteritems():
            init_program.global_block()._clone_variable(var)
            initializer(var, init_program.global_block())
        exe = Executor(place)
        exe.run(program=init_program, scope=self.scope)

    def has(self, attr_name):
        return attr_name in self._attrs

    def set(self, attr_name, attr):
        if not has(attr_name):
            self._attrs[attr_name] = attr
        else:
            raise ValueError("{} attr already set in the graph.".format(
                attr_name))

    def get(self, attr_name):
        if has(attr_name):
            return self._attrs[attr_name]
        else:
            raise ValueError("{} attr not registered in the graph.".format(
                attr_name))

    def all_parameters(self):
        return self.program.block(0).all_parameters()

    def create_parameter(self, *args, **kwargs):
        return self.program.block(0).create_parameter(*args, **kwargs)

    def all_vars(self):
        for each_var in list(self.program.block(0).vars.values()):
            yield each_var

    def vars_map(self):
        return self.program.block(0).vars

    def all_ops(self):
        return self.program.block(0).ops

    def index(self, op):
        return self.program.block(0).ops.index(op)

    def var(self, name):
        return self.program.block(0).var(name)

    def create_var(self, *args, **kwargs):
        return self.program.block(0).create_var(*args, **kwargs)

    def remove_var(self, name):
        self.program.block(0)._remove_var(name)

    def insert_op(self, index, *args, **kwargs):
        return self.program.block(0)._insert_op(index=index, *args, **kwargs)

    def prepend_op(self, *args, **kwargs):
        return self.program.block(0)._prepend_op(*args, **kwargs)

    def remove_op(self, index):
        self.program.block(0)._remove_op(index)

    def prune(self, feeds, fetches):
        if not isinstance(feeds, Iterable):
            feeds = [feeds]

        if not isinstance(fetches, Iterable):
            fetches = [fetches]

        program = self.program._prune(fetches)

        feeds = [
            feed.name if isinstance(feed, Variable) else feed for feed in feeds
        ]
        fetches = [
            fetch.name if isinstance(fetch, Variable) else fetch
            for fetch in fetches
        ]

        in_nodes = OrderedDict([(key, value)
                                for key, value in self.in_nodes.items()
                                if value in feeds])
        out_nodes = OrderedDict([(key, value)
                                 for key, value in self.out_nodes.items()
                                 if value in fetches])
        return ImitationGraph(program, self.scope, in_nodes, out_nodes)

    def get_var(self, var_name):
        return self.program.global_block().var(var_name)

    def clone(self, for_test=False):
        return ImitationGraph(
            self.program.clone(for_test), self.scope,
            copy.deepcopy(self.in_nodes), copy.deepcopy(self.out_nodes))

    def merge(self, graph):
        for var in graph.program.list_vars():
            self.program.global_block()._clone_variable(var)
        for op in graph.all_ops():
            inputs = {}
            outputs = {}
            attrs = {}
            for input_name in op.input_names:
                inputs[input_name] = [
                    self.get_var(in_var_name)
                    for in_var_name in op.input(input_name)
                ]
            for output_name in op.output_names:
                outputs[output_name] = [
                    self.get_var(out_var_name)
                    for out_var_name in op.output(output_name)
                ]
            for attr_name in op.attr_names:
                attrs[attr_name] = op.attr(attr_name)
            self.program.global_block().append_op(
                type=op.type, inputs=inputs, outputs=outputs, attrs=attrs)

    def ops(self):
        return self.program.global_block().ops

    def program(self):
        return self.program

    def pre_ops(self, op):
        ops = []
        in_var_names = []
        for input_name in op.input_names:
            in_var_names += op.input(input_name)
        for p in self.ops():
            for out_name in p.output_names:
                for var_name in p.output(out_name):
                    if var_name in in_var_names:
                        ops.append(p)
        return ops

    def next_ops(self, op):
        ops = []
        out_var_names = []
        for o in op.output_names:
            out_var_names += op.output(o)
        for p in self.ops():
            for input_name in p.input_names:
                for var_name in p.input(input_name):
                    if var_name in out_var_names:
                        ops.append(p)
        return ops

    def get_ops_by_param(self, param):
        ops = []
        if isinstance(param, Variable):
            param = param.name
        for op in self.ops():
            for name in op.input_names:
                if param in op.input(name):
                    ops.append(op)
        return ops

    def get_param_by_op(self, op):
        params = []
        for in_name in op.input_names:
            for var_name in op.input(in_name):
                var = self.get_var(var_name)
                if isinstance(var, Parameter):
                    params.append(var)
        return params

    def flops(self):
        ret = 0
        b_vars = {}
        for var in self.program.list_vars():
            b_vars[var.name] = var
        for op in self.all_ops():
            if op.type in ['conv2d', 'depthwise_conv2d', 'mul']:
                _, _, _, flop = _count_shape_params_flops(b_vars, op)
                ret += flop
        return ret

    def numel_params(self):
        ret = 0
        for param in self.all_parameters():
            ret += np.product(param.shape)
        return ret

    def serialize(self):
        data = {}
        data['program'] = self.program
        data['in_nodes'] = self.in_nodes
        data['out_nodes'] = self.out_nodes
        data['attrs'] = self._attrs
        return pickle.dumps(data)

    def deserialize(self, s):
        data = pickle.loads(s)
        self.program = data['program']
        self.in_nodes = data['in_nodes']
        self.out_nodes = data['out_nodes']
        self._attrs = data['attrs']

    def get_optimize_graph(self, optimizer, place):
        graph = self.clone()
        startup_program = Program()
        with program_guard(
                main_program=graph.program, startup_program=startup_program):
            target_name = None
            if 'loss' in graph.out_nodes:
                target_name = graph.out_nodes['loss']
            elif 'cost' in graph.out_nodes:
                target_name = graph.out_nodes['cost']
            target = graph.get_var(target_name)
            optimizer.minimize(target)

        exe = Executor(place)
        exe.run(program=startup_program, scope=self.scope)
        return graph


class IRGraph(Graph):
    pass


def save_persistables(graph, path, exe):
    io.save_persistables(exe.exe, path, main_program=graph.program)


def load_persistables(graph, path, exe):
    def if_exist(var):
        return os.path.exists(os.path.join(path, var.name))

    io.load_vars(exe.exe, path, main_program=graph.program, predicate=if_exist)
    update_param_shape(graph)
    update_depthwise_conv(graph)


def update_param_shape(graph):
    for param in graph.all_parameters():
        tensor_shape = np.array(graph.scope.find_var(param.name).get_tensor(
        )).shape
        param.desc.set_shape(tensor_shape)


def infer_shape(graph):
    for op in graph.all_ops():
        if op.type != 'conditional_block':
            op.desc.infer_shape(op.block.desc)


def update_depthwise_conv(graph):
    for op in graph.all_ops():
        if op.type == 'depthwise_conv2d':
            op.desc._set_attr('groups',
                              graph.get_var(op.input('Filter')[0]).shape[0])


def save_inference_graph_model(dirname,
                               feeded_var_names,
                               target_var_names,
                               place,
                               graph=None,
                               model_filename=None,
                               params_filename=None,
                               export_for_deployment=True):
    target_vars = None
    if target_var_names:
        target_vars = [graph.var(name) for name in target_var_names]
    exe = Executor(place)
    save_inference_model(dirname, feeded_var_names, target_vars, exe,
                         graph.program, model_filename, params_filename,
                         export_for_deployment)


def load_inference_graph_model(dirname,
                               place,
                               model_filename=None,
                               params_filename=None,
                               pserver_endpoints=None):
    exe = Executor(place)
    return load_inference_model(dirname, exe, model_filename, params_filename,
                                pserver_endpoints)


def _count_shape_params_flops(b_vars, one_op):
    '''
    Args:
        b_vars: all vars of one block
        one_op: one operator to count
    Returns:
        in_data_shape: one operator's input data shape
        out_data_shape: one operator's output data shape
        PARAMs: one operator's PARAMs 
        FLOPs: : one operator's FLOPs
    '''
    if one_op.type in ['conv2d', 'depthwise_conv2d']:
        k_arg_shape = b_vars[one_op.input("Filter")[0]].shape
        in_data_shape = b_vars[one_op.input("Input")[0]].shape
        out_data_shape = b_vars[one_op.output("Output")[0]].shape
        c_out, c_in, k_h, k_w = k_arg_shape
        _, c_out_, data_h, data_w = out_data_shape
        #        assert c_out == c_out_, 'shape error!'
        k_groups = one_op.attr("groups")
        kernel_ops = k_h * k_w * (c_in / k_groups)
        # keras's conv use bias defaultly
        # bias_ops = 0 if one_op.input("Bias") == [] else 1
        bias_ops = 0  # for test
        PARAMs = c_out * (kernel_ops + bias_ops)
        FLOPs = 2 * data_h * data_w * c_out * (kernel_ops + bias_ops)

    elif one_op.type == 'pool2d':
        in_data_shape = b_vars[one_op.input("X")[0]].shape
        out_data_shape = b_vars[one_op.output("Out")[0]].shape
        _, c_out, data_h, data_w = out_data_shape
        k_size = one_op.attr("ksize")
        PARAMs = 0
        FLOPs = data_h * data_w * c_out * (k_size[0]**2)

    elif one_op.type == 'mul':
        k_arg_shape = b_vars[one_op.input("Y")[0]].shape
        in_data_shape = b_vars[one_op.input("X")[0]].shape
        out_data_shape = b_vars[one_op.output("Out")[0]].shape
        # TODO: fc has mul ops
        # add attr to mul op, tell us whether it belongs to 'fc'
        # this's not the best way
        if 'fc' not in one_op.output("Out")[0]:
            return None
        k_in, k_out = k_arg_shape
        # bias in sum op
        PARAMs = k_in * k_out + 1
        FLOPs = k_in * k_out

    elif one_op.type in ['relu', 'sigmoid']:
        in_data_shape = b_vars[one_op.input("X")[0]].shape
        out_data_shape = b_vars[one_op.output("Out")[0]].shape
        _, c_in, data_h, data_w = in_data_shape
        PARAMs = 0
        FLOPs = data_h * data_w * c_in

    elif one_op.type == 'batch_norm':
        in_data_shape = b_vars[one_op.input("X")[0]].shape
        out_data_shape = b_vars[one_op.output("Y")[0]].shape
        _, c_in, data_h, data_w = in_data_shape
        # gamma, beta, mean, std
        PARAMs = c_in * 4
        FLOPs = data_h * data_w * c_in

    else:
        return None

    return in_data_shape, out_data_shape, PARAMs, FLOPs
