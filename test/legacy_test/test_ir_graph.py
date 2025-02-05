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

from paddle import base


class TestIRGraph(unittest.TestCase):
    """
    TODO(fc500110): `resolve_hazard` api will be tested when it can be used.
    """

    def test_nodes(self):
        graph = build_graph()
        self.assertTrue(
            {node.name() for node in graph.nodes()}
            == {"x1", "x2", "out", "sum"}
        )

    def test_has_set_get(self):
        graph = build_graph()
        for attr_name in ["int", "float", "string"]:
            self.assertFalse(graph.has(attr_name))
        graph.set("int", 1)
        graph.set("float", 0.5)
        graph.set("string", "string")
        for attr_name in ["int", "float", "string"]:
            self.assertTrue(graph.has(attr_name))

        self.assertTrue(graph.get_int("int") == 1)
        self.assertTrue(graph.get_float("float") == 0.5)
        self.assertTrue(graph.get_string("string") == "string")

    def test_erase(self):
        graph = build_graph()
        graph.set("test", 0)
        self.assertTrue(graph.has("test"))
        graph.erase("test")
        self.assertFalse(graph.has("test"))

    def test_create_var_node(self):
        prog = base.core.ProgramDesc()
        block = prog.block(0)
        shape = [10, 20]
        x1 = block.var(b'x1')
        x1.set_type(base.core.VarDesc.VarType.DENSE_TENSOR)
        x1.set_shape(shape)
        graph = base.core.Graph(prog)
        node = graph.create_var_node(x1)
        self.assertTrue(node.node_type() == base.core.Node.Type.Variable)

    def test_create_op_node(self):
        prog = base.core.ProgramDesc()
        block = prog.block(0)
        sum_op_desc = block.append_op()
        graph = base.core.Graph(prog)
        node = graph.create_op_node(sum_op_desc)
        self.assertTrue(node.node_type() == base.core.Node.Type.Operation)

    def test_create_control_dep_var(self):
        graph = build_graph()
        name = f"__control_var@{len(graph.nodes())}"
        node = graph.create_control_dep_var()
        self.assertTrue(node.name() == name)

    def test_create_empty_node(self):
        prog = base.core.ProgramDesc()
        graph = base.core.Graph(prog)
        n1 = graph.create_empty_node('x', base.core.Node.Type.Operation)
        self.assertTrue(n1.name() == 'x')
        n2 = graph.create_empty_node('y', base.core.Node.Type.Variable)
        self.assertTrue(n2.name() == 'y')

    def test_release_nodes(self):
        graph = build_graph()
        nodes = graph.release_nodes()
        self.assertTrue(len(graph.nodes()) == 0)
        self.assertTrue(
            {node.name() for node in nodes} == {"x1", "x2", "out", "sum"}
        )

    def test_remove_node(self):
        graph = build_graph()
        nodes = graph.nodes()
        for node in nodes:
            if node.name() == "sum":
                break
        self.assertTrue(
            {node.name() for node in nodes} == {"x1", "x2", "out", "sum"}
        )
        nodes.remove(node)
        self.assertTrue({node.name() for node in nodes} == {"x1", "x2", "out"})

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
    prog = base.core.ProgramDesc()
    block = prog.block(0)

    shape = [10, 20]

    # prepare input/output
    x1 = block.var(b'x1')
    x1.set_type(base.core.VarDesc.VarType.DENSE_TENSOR)
    x1.set_shape(shape)
    x2 = block.var(b'x2')
    x2.set_type(base.core.VarDesc.VarType.DENSE_TENSOR)
    x2.set_shape(shape)

    out = block.var(b'out')
    out.set_type(base.core.VarDesc.VarType.DENSE_TENSOR)

    sum_op_desc = block.append_op()
    sum_op_desc.set_type("sum")
    sum_op_desc.set_input("X", ["x1", "x2"])
    sum_op_desc.set_output("Out", ["out"])

    sum_op_desc.check_attrs()
    sum_op_desc.infer_shape(block)
    graph = base.core.Graph(prog)
    return graph


if __name__ == "__main__":
    unittest.main()
