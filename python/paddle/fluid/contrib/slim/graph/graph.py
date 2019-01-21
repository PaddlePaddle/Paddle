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
from __future__ import print_function
import os
import subprocess
from ....framework import Program
from ....framework import Block
from .... import core

__all__ = ['Graph', 'ImitationGraph', 'IRGraph', 'PyGraph']


class PyGraph(object):
    """
    PyGraph uses core.Graph as the delegation to accomplish the manipulation.
    """

    def __init__(self, graph, for_test=False):
        """
        Construct the PyGraph using core.Graph.
        Args:
            graph(core.Graph): C++ Graph.
            for_test(bool): True for the test graph and false for the train graph.
        """
        assert isinstance(
            graph, core.Graph), 'graph must be the instance of core.Graph.'
        self.graph = graph
        self.for_test = for_test

    def is_test(self):
        return self.for_test

    def all_parameters(self):
        param_nodes = set()
        for node in self.graph.nodes():
            if node.is_var() and node.var() is not None and node.var(
            ).persistable():
                param_nodes.add(node)
        return param_nodes

    def all_vars(self):
        return {node for node in self.graph.nodes() if node.is_var()}

    def all_ops(self):
        return {node for node in self.graph.nodes() if node.is_op()}

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
        for input_name, var_nodes in inputs.iteritems():
            if not isinstance(var_nodes, list):
                var_nodes = [var_nodes]
            op_desc.set_input(input_name,
                              [var_node.name() for var_node in var_nodes])
        for output_name, var_nodes in outputs.iteritems():
            if not isinstance(var_nodes, list):
                var_nodes = [var_nodes]
            op_desc.set_output(output_name,
                               [var_node.name() for var_node in var_nodes])
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

    def safe_remove_nodes(self, remove_nodes):
        if not isinstance(remove_nodes, set):
            remove_nodes = set(remove_nodes)
        core.graph_safe_remove_nodes(self.graph, remove_nodes)

    def draw(self, save_path, name, marked_nodes=None):
        def _convert_to_pdf(dot_file_path):
            pdf_save_path = os.path.splitext(dot_file_path)[0] + '.pdf'
            exited_code = subprocess.call('dot -Tpdf ' + dot_file_path \
                            + ' -o ' + pdf_save_path, shell=True)
            if exited_code != 0:
                print('The dot command is needed for creating pdf files.')
                print('The {} is saved as the dot filetype.'.format(
                    dot_file_path))

        remove_ctr_vars = set()
        ops_num = 0
        for node in self.graph.nodes():
            if node.is_ctrl_var():
                remove_ctr_vars.add(node)
            elif node.is_op():
                ops_num += 1
        print('Total ops num = {}.'.format(ops_num))
        self.safe_remove_nodes(remove_ctr_vars)
        if marked_nodes is not None:
            if not isinstance(marked_nodes, set):
                marked_nodes = set(marked_nodes)
            marked_nodes = marked_nodes - remove_ctr_vars
            if self.graph.has('__graphviz__marked_node__'):
                self.graph.erase('__graphviz__marked_node__')
            self.graph.set('__graphviz__marked_node__', marked_nodes)
        viz_dot_path = os.path.join(save_path, name) + '.dot'
        viz_pass = core.get_pass('graph_viz_pass')
        viz_pass.set_str('graph_viz_path', viz_dot_path)
        viz_pass.apply(self.graph)
        _convert_to_pdf(viz_dot_path)

    def to_program(self):
        convert_pass = core.get_pass('graph_to_program_pass')
        convert_pass.set_program('program', Program().desc)
        convert_pass.apply(self.graph)
        desc = convert_pass.get_program('program')
        program = Program.construct_from_desc(desc)
        return program


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


class IRGraph(Graph):
    pass
