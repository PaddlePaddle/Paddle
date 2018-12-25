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

from ....framework import Program
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

    def create_var(self, name, type, shape, dtype):
        """
        Create a var in the graph.
        """
        pass

    def remove_var(self, name):
        """
        Remove a var from the graph by the given name.
        """
        pass

    def insert_op(self, idx, type, attrs, inputs, outputs):
        """
        Insert an operation before the idx op.
        """
        pass

    def remove_op(self, idx):
        """
        Remove the index idx operation.
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
    def __init__(self, program=None):
        super(ImitationGraph, self).__init__()
        self.program = Program() if program is None else program

    def all_parameters(self):
        return self.program.global_block().all_parameters()

    def create_parameter(self, *args, **kwargs):
        return self.program.global_block().create_parameter(*args, **kwargs)

    def all_vars(self):
        return self.program.list_vars()

    def vars_map(self):
        return self.program.blocks[0].vars

    def all_ops(self):
        return self.program.blocks[0].ops

    def index(self, op):
        return self.program.blocks[0].ops.index(op)

    def var(self, name):
        return self.program.blocks[0].var(name)

    def create_var(self, name, type, shape, dtype):
        return self.program.blocks[0].create_var(
            name=name, type=type, shape=shape, dtype=dtype)

    def remove_var(self, name):
        self.program.blocks[0]._remove_var(name)

    def insert_op(self, idx, type, attrs, inputs, outputs):
        return self.program.blocks[0]._insert_op(
            idx, type=type, attrs=attrs, inputs=inputs, outputs=outputs)

    def remove_op(self, idx):
        self.program.blocks[0]._remove_op(idx)

    def clone(self, for_test=False):
        return ImitationGraph(self.program.clone(for_test))

    def prune(self, feeds, fetches):
        return ImitationGraph(self.program._prune(fetches))


class IRGraph(Graph):
    pass


def save_inference_graph_model(dirname,
                               feeded_var_names,
                               target_var_names,
                               executor,
                               graph=None,
                               model_filename=None,
                               params_filename=None,
                               export_for_deployment=True):
    target_vars = None
    if target_var_names:
        target_vars = [graph.var(name) for name in target_var_names]

    save_inference_model(dirname, feeded_var_names, target_vars, executor.exe,
                         graph.program, model_filename, params_filename,
                         export_for_deployment)


def load_inference_graph_model(dirname,
                               executor,
                               model_filename=None,
                               params_filename=None,
                               pserver_endpoints=None):
    return load_inference_model(dirname, executor.exe, model_filename,
                                params_filename, pserver_endpoints)
