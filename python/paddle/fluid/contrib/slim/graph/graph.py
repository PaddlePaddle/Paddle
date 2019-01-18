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
from ....framework import Block
from .... import core

__all__ = ['Graph', 'ImitationGraph', 'PyGraph']


class PyGraph(object):
    """
    PyGraph uses core.Graph as the delegation to accomplish the manipulation.
    """

    def __init__(self, graph):
        assert isinstance(
            graph, core.Graph), 'graph must be the instance of core.Graph.'
        self.graph = graph

    def all_parameters(self):
        params = []
        for node in self.graph.nodes():
            if node.is_var() and node.var().persistable():
                params.append(node)
        return params

    def all_vars(self):
        return [node for node in self.graph.nodes() if node.is_var()]

    def all_ops(self):
        return [node for node in self.graph.nodes() if node.is_op()]

    def create_param_node(self, name, var_type, shape, var_dtype):
        var_desc = core.VarDesc(name)
        var_desc.set_type(var_type)
        var_desc.set_shape(shape)
        var_desc.set_dtype(var_dtype)
        var_desc.set_persistable(True)
        return self.graph.create_var_node(var_desc)

    def create_var_node(self, name, var_type, shape, var_dtype):
        var_desc = core.VarDesc(name)
        var_desc.set_type(var_type)
        var_desc.set_shape(shape)
        var_desc.set_dtype(var_dtype)
        return self.graph.create_var_node(var_desc)

    def create_var_node_from_desc(self, var_desc):
        return self.graph.create_var_node(var_desc)

    def create_op_node(self, op_type, attrs, inputs, outputs):
        op_desc = core.OpDesc()
        op_desc.set_type(op_type)
        for attr, value in attrs.iteritems():
            self._update_desc_attr(op_desc, attr, value)
        for input_name, var_node in inputs.iteritems():
            op_desc.set_input(input_name, [var_node.name()])
        for output_name, var_node in outputs.iteritems():
            op_desc.set_output(output_name, [var_node.name()])
        return self.graph.create_op_node(op_desc)

    def create_op_node_from_desc(self, op_desc):
        return self.graph.create_op_node(op_desc)

    def _update_desc_attr(self, desc, name, val):
        """
        Update the value of desc's attribute by attribute's name.
        """
        if isinstance(val, Block):
            desc.set_block_attr(name, val.desc)
        elif isinstance(val, list) and val and all(
                isinstance(v, Block) for v in val):
            desc.set_blocks_attr(name, [v.desc for v in val])
        elif isinstance(val, core.BlockDesc) or \
                isinstance(val, core.ProgramDesc):
            desc.set_serialized_attr(name, val.serialize_to_string())
        else:
            desc._set_attr(name, val)


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


class ImitationGraph(Graph):
    def __init__(self, program=None):
        super(ImitationGraph, self).__init__()
        self.program = Program() if program is None else program

    def all_parameters(self):
        return self.program.global_block().all_parameters()
