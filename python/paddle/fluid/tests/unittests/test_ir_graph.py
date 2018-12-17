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

import unittest
import six
from paddle import fluid


class TestIRGraph(unittest.TestCase):
    def test_nodes(self):
        prog = fluid.core.ProgramDesc()
        block = prog.block(0)

        shape = [10, 20]

        # prepare input/output
        x1 = block.var(six.b("x1"))
        x1.set_type(fluid.core.VarDesc.VarType.LOD_TENSOR)
        x1.set_shape(shape)
        x2 = block.var(six.b("x2"))
        x2.set_type(fluid.core.VarDesc.VarType.LOD_TENSOR)
        x2.set_shape(shape)

        out = block.var(six.b("out"))
        out.set_type(fluid.core.VarDesc.VarType.LOD_TENSOR)

        sum_op_desc = block.append_op()
        sum_op_desc.set_type("sum")
        sum_op_desc.set_input("X", ["x1", "x2"])
        sum_op_desc.set_output("Out", ["out"])

        sum_op_desc.check_attrs()
        sum_op_desc.infer_shape(block)
        graph = fluid.core.Graph(prog)
        self.assertTrue({node.name() for node in graph.nodes()} == {"x1", "x2", "out", "sum"})

    def test_get(self):
        pass

    def test_set(self):
        pass

    def test_set_not_owned(self):
        pass

    def test_erase(self):
        pass

    def test_create_var_node(self):
        #graph = build_graph()
        pass

    def test_create_op_node(self):
        pass

    def test_create_control_dep_var(self):
        pass

    def test_create_empty_node(self):
        prog = fluid.core.ProgramDesc()
        graph = fluid.core.Graph(prog)
        n1 = graph.create_empty_node('x', fluid.core.NodeType.Operation)
        self.assertTrue(n1.name() == 'x')
        n2 = graph.create_empty_node('y', fluid.core.NodeType.Variable)
        self.assertTrue(n2.name() == 'y')


    def test_release_nodes(self):
        graph = build_graph()
        nodes = graph.release_nodes()
        self.assertTrue(len(graph.nodes()) == 0)
        self.assertTrue({node.name() for node in nodes} == {"x1", "x2", "out", "sum"})

    def test_remove_node(self):
        pass

    def test_retrieve_node(self):
        graph = build_graph()
        nodes = []
        for i in range(len(graph.nodes())):
            nodes.append(graph.retrieve_node(i))

        for node in nodes:
            self.assertTrue(node in graph.nodes())

    def resolve_hazard(self):
        pass


def build_graph():
    prog = fluid.core.ProgramDesc()
    block = prog.block(0)

    shape = [10, 20]

    # prepare input/output
    x1 = block.var(six.b("x1"))
    x1.set_type(fluid.core.VarDesc.VarType.LOD_TENSOR)
    x1.set_shape(shape)
    x2 = block.var(six.b("x2"))
    x2.set_type(fluid.core.VarDesc.VarType.LOD_TENSOR)
    x2.set_shape(shape)

    out = block.var(six.b("out"))
    out.set_type(fluid.core.VarDesc.VarType.LOD_TENSOR)

    sum_op_desc = block.append_op()
    sum_op_desc.set_type("sum")
    sum_op_desc.set_input("X", ["x1", "x2"])
    sum_op_desc.set_output("Out", ["out"])

    sum_op_desc.check_attrs()
    sum_op_desc.infer_shape(block)
    graph = fluid.core.Graph(prog)
    return graph

if __name__ == "__main__":
    unittest.main()
