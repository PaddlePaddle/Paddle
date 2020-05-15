#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import textwrap
import gast
from paddle.fluid.dygraph.dygraph_to_static.ifelse_transformer import get_name_ids
from paddle.fluid.dygraph.dygraph_to_static.ifelse_transformer import IfConditionVisitor
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import NodeVarType
from paddle.fluid.dygraph.dygraph_to_static.utils import IsControlFlowVisitor


class TestGetNameIds(unittest.TestCase):
    """
    Test for parsing the ast.Name list from the ast.Nodes
    """

    def setUp(self):
        self.source = """
          def test_fn(x):
            return x+1
        """
        self.all_name_ids = {'x': [gast.Param()]}

    def test_get_name_ids(self):
        source = textwrap.dedent(self.source)
        root = gast.parse(source)
        all_name_ids = get_name_ids([root])
        self.assertDictEqual(
            self.transfer_dict(self.all_name_ids),
            self.transfer_dict(all_name_ids))

    def transfer_dict(self, name_ids_dict):
        new_dict = {}
        for name, ctxs in name_ids_dict.items():
            new_dict[name] = [type(ctx) for ctx in ctxs]
        return new_dict


class TestGetNameIds2(TestGetNameIds):
    def setUp(self):
        self.source = """
          def test_fn(x, y):
            a = 1
            x = y + a
            if x > y:
               z = x * x
               z = z + a
            else:
               z = y * y
            return z
        """
        self.all_name_ids = {
            'x': [
                gast.Param(), gast.Store(), gast.Load(), gast.Load(),
                gast.Load()
            ],
            'a': [gast.Store(), gast.Load(), gast.Load()],
            'y': [
                gast.Param(),
                gast.Load(),
                gast.Load(),
                gast.Load(),
                gast.Load(),
            ],
            'z': [
                gast.Store(),
                gast.Load(),
                gast.Store(),
                gast.Store(),
            ]
        }


class TestGetNameIds3(TestGetNameIds):
    def setUp(self):
        self.source = """
          def test_fn(x, y):
            z = 1
            if x > y:
               z = x * x
               z = z + y
            return z
        """
        self.all_name_ids = {
            'x': [
                gast.Param(),
                gast.Load(),
                gast.Load(),
                gast.Load(),
            ],
            'y': [
                gast.Param(),
                gast.Load(),
                gast.Load(),
            ],
            'z': [
                gast.Store(),
                gast.Store(),
                gast.Load(),
                gast.Store(),
            ]
        }


class TestIsControlFlowIf(unittest.TestCase):
    def check_false_case(self, code):
        code = textwrap.dedent(code)
        node = gast.parse(code)
        node_test = node.body[0].value

        if_visitor = IfConditionVisitor(node_test)
        self.assertFalse(if_visitor.is_control_flow())
        # No transformation will be applied.
        new_node, assign_nodes = if_visitor.transform()
        self.assertTrue(new_node == node_test)
        self.assertTrue(len(assign_nodes) == 0)

    def test_expr(self):
        # node is not ast.Compare
        self.check_false_case("a+b")

    def test_expr2(self):
        # x is a Tensor.
        node = gast.parse("a + x.numpy()")
        node_test = node.body[0].value

        if_visitor = IfConditionVisitor(node_test)
        self.assertTrue(if_visitor.is_control_flow())
        # No transformation will be applied.
        new_node, assign_nodes = if_visitor.transform()
        self.assertTrue(len(assign_nodes) == 0)

    def test_is_None(self):
        self.check_false_case("x is None")

    def test_is_None2(self):
        self.check_false_case("fluid.layers.sum(x) is None")

    def test_is_None3(self):
        self.check_false_case("fluid.layers.sum(x).numpy() != None")

    def test_is_None4(self):
        node = gast.parse("fluid.layers.sum(x) and 2>1")
        node_test = node.body[0].value

        if_visitor = IfConditionVisitor(node_test)
        self.assertTrue(if_visitor.is_control_flow())
        # Transformation result:
        # bool_tensor_0 = fluid.layers.cast(x=fluid.layers.sum(x), dtype='bool')
        # bool_tensor_1 = fluid.layers.fill_constant(shape=[1], dtype='bool', value=bool(2 > 1))
        # logic_and_0 = fluid.layers.logical_and(x=bool_tensor_0, y=bool_tensor_1)

        new_node, assign_nodes = if_visitor.transform()
        self.assertTrue(len(assign_nodes) == 3)

    def test_if(self):
        node = gast.parse("x.numpy()[1] > 1")
        node_test = node.body[0].value

        if_visitor = IfConditionVisitor(node_test)
        self.assertTrue(if_visitor.is_control_flow())
        # No transformation will be applied.
        new_node, assign_nodes = if_visitor.transform()
        self.assertTrue(len(assign_nodes) == 0)

    def test_if_with_and(self):
        node = gast.parse("x and 1 < x.numpy()[1]")
        node_test = node.body[0].value

        if_visitor = IfConditionVisitor(node_test)
        self.assertTrue(if_visitor.is_control_flow())
        # No transformation will be applied.
        new_node, assign_nodes = if_visitor.transform()
        self.assertTrue(isinstance(new_node, gast.Name))
        self.assertTrue(len(assign_nodes) == 2)

    def test_if_with_or(self):
        node = gast.parse("1 < fluid.layers.sum(x).numpy()[2] or x+y < 0")
        node_test = node.body[0].value

        if_visitor = IfConditionVisitor(node_test)
        self.assertTrue(if_visitor.is_control_flow())
        # No transformation will be applied.
        new_node, assign_nodes = if_visitor.transform()
        self.assertTrue(isinstance(new_node, gast.Name))
        self.assertTrue(len(assign_nodes) == 2)

    def test_shape(self):
        code = """
            def foo(x):
                batch_size = fluid.layers.shape(x)
                if batch_size[0] > 16:
                    x = x + 1
                return x
        """
        code = textwrap.dedent(code)
        node = gast.parse(code)
        static_analysis_visitor = StaticAnalysisVisitor(node)
        test_node = node.body[0].body[1].test
        if_visitor = IfConditionVisitor(test_node, static_analysis_visitor)
        self.assertTrue(if_visitor.is_control_flow())
        # No transformation will be applied.
        new_node, assign_nodes = if_visitor.transform()
        self.assertTrue(new_node == test_node)
        self.assertTrue(len(assign_nodes) == 0)

    def test_shape_with_andOr(self):
        code = """
            def foo(x):
                batch_size = fluid.layers.shape(x)
                if x is not None and batch_size[0] > 16 or 2 > 1:
                    x = x + 1
                return x
        """
        code = textwrap.dedent(code)
        node = gast.parse(code)
        static_analysis_visitor = StaticAnalysisVisitor(node)
        test_node = node.body[0].body[1].test
        if_visitor = IfConditionVisitor(test_node, static_analysis_visitor)
        self.assertTrue(if_visitor.is_control_flow())

        new_node, assign_nodes = if_visitor.transform()
        # transformation result:
        # bool_tensor_0 = fluid.layers.fill_constant(shape=[1], dtype='bool', value=bool(x is not None))
        # logic_and_0 = fluid.layers.logical_and(x=bool_tensor_0, y=batch_size[0] > 16)
        # bool_tensor_1 = fluid.layers.fill_constant(shape=[1], dtype='bool', value=bool(2 > 1))
        # logic_or_0 = fluid.layers.logical_or(x=logic_and_0, y=bool_tensor_1)

        self.assertTrue(isinstance(new_node, gast.Name))
        self.assertTrue(len(assign_nodes) == 4)

    def test_paddle_api(self):
        code = """
            def foo(x):
                if fluid.layers.shape(x)[0] > 16:
                    x = x + 1
                return x
        """
        code = textwrap.dedent(code)
        node = gast.parse(code)
        static_analysis_visitor = StaticAnalysisVisitor(node)
        test_node = node.body[0].body[0].test
        if_visitor = IfConditionVisitor(test_node, static_analysis_visitor)
        self.assertTrue(if_visitor.is_control_flow())

        # No transformation will be applied.
        new_node, assign_nodes = if_visitor.transform()
        self.assertTrue(new_node == test_node)
        self.assertTrue(len(assign_nodes) == 0)

    def test_paddle_api_with_andOr(self):
        code_or = """
            def foo(x):
                if 2 > 1 and fluid.layers.shape(x)[0] > 16 or x is not None :
                    x = x + 1
                return x
        """

        code_and = """
            def foo(x):
                if 2 > 1 and fluid.layers.shape(x)[0] > 16 and x is not None :
                    x = x + 1
                return x
        """
        for code in [code_or, code_and]:
            code = textwrap.dedent(code)
            node = gast.parse(code)
            static_analysis_visitor = StaticAnalysisVisitor(node)
            test_node = node.body[0].body[0].test
            if_visitor = IfConditionVisitor(test_node, static_analysis_visitor)
            self.assertTrue(if_visitor.is_control_flow())

            new_node, assign_nodes = if_visitor.transform()
            # Transformation result:
            # bool_tensor_0 = fluid.layers.fill_constant(shape=[1], dtype='bool', value=bool(2 > 1))
            # bool_tensor_1 = fluid.layers.fill_constant(shape=[1], dtype='bool', value=bool(x is not None))
            # logic_and_0 = fluid.layers.logical_and(x=bool_tensor_0, y=fluid.layers.shape(x)[0] > 16)

            # logic_and_1 = fluid.layers.logical_and(x=logic_and_0, y=bool_tensor_1)  for code_and
            # logic_or_0= fluid.layers.logical_or(x=logic_and_0, y=bool_tensor_1)  for code_and

            self.assertTrue(isinstance(new_node, gast.Name))
            self.assertTrue(len(assign_nodes) == 4)

    def test_with_node_var_type_map(self):
        node = gast.parse("x > 1")
        node_test = node.body[0].value

        # if x is a Tensor
        node_var_type_map = {"x": {NodeVarType.TENSOR}}
        visitor = IsControlFlowVisitor(
            node_test, node_var_type_map=node_var_type_map)
        self.assertTrue(visitor.transform())

        # if x is not a Tensor
        node_var_type_map = {"x": {NodeVarType.NUMPY_NDARRAY}}
        visitor = IsControlFlowVisitor(
            node_test, node_var_type_map=node_var_type_map)
        self.assertFalse(visitor.transform())

    def test_raise_error(self):
        node = "a + b"
        with self.assertRaises(Exception) as e:
            self.assertRaises(TypeError, IfConditionVisitor(node))
        self.assertTrue(
            "Type of input node should be gast.AST" in str(e.exception))


if __name__ == '__main__':
    unittest.main()
