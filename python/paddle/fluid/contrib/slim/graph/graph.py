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
from ....framework import Program
import copy
from ....io import save_inference_model, load_inference_model

__all__ = [
    'Graph', 'ImitationGraph', 'IRGraph', 'save_inference_graph_model',
    'load_inference_graph_model'
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
        self.scope = scope
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self._attrs = collections.OrderedDict()

    def init_vars(self, need_inited, place):
        init_program = fluid.Program()
        for var, initializer in need_inited.iteritems():
            init_program.global_block()._clone_variable(var)
            initializer(var, init_program.global_block())
        exe = fuid.executor.Executor(place)
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

    def clone(self, for_test=False):
        return ImitationGraph(self.program.clone(for_test))

    def prune(self, feeds, fetches):
        return ImitationGraph(self.program._prune(fetches))

    def get_var(self, var_name):
        return self.program.global_block().var(var_name)

    def clone(self):
        return ImitationGraph(self.program.clone(), self.scope,
                              copy.deepcopy(self.in_nodes),
                              copy.deepcopy(self.out_nodes))

    def ops(self):
        return self.program.global_block().ops

    def program(self):
        return self.program

    def pre_ops(self, op):
        ops = []
        in_names = []
        for i in op.input_names:
            in_names += op.input(i)
        for p in self.ops():
            for out in p.output_names:
                for o in p.output(out):
                    if o in in_names:
                        ops.append(p)
        return ops


class IRGraph(Graph):
    pass


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
    exe = fluid.executor.Executor(place)
    save_inference_model(dirname, feeded_var_names, target_vars, exe,
                         graph.program, model_filename, params_filename,
                         export_for_deployment)


def load_inference_graph_model(dirname,
                               place,
                               model_filename=None,
                               params_filename=None,
                               pserver_endpoints=None):
    exe = fluid.executor.Executor(place)
    return load_inference_model(dirname, exe, model_filename, params_filename,
                                pserver_endpoints)
