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

import astor
import gast

from paddle.fluid import unique_name
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.utils import is_dygraph_api, is_to_variable
from paddle.fluid.dygraph.dygraph_to_static.utils import to_assign_node, to_static_ast, update_args_of_func
from paddle.fluid.dygraph.dygraph_to_static.utils import dygraph_class_to_static_api


class BasicApiTransformer(gast.NodeTransformer):
    """
    Class to transform basic API from dygraph to static graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of BasicApiTransformer."

        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.class_node_dict = {}

        # Used for transformation of data feed
        self.feed_name_to_arg_id = {}
        self.name_to_tensor_shape = {}

    def transform(self):
        self.visit(self.root)
        return self.wrapper_root

    def visit_Assign(self, node):
        if self._update_class_node_dict(node):
            return None

        for child_node in gast.walk(node.value):
            if isinstance(child_node, gast.Call):
                self._visit_Call(child_node)
        return node

    def visit_Expr(self, node):
        value_node = node.value
        for child_node in gast.walk(value_node):
            if isinstance(child_node, gast.Call):
                # TODO(liym27):
                #  Considers that a dygraph api which modifies the input or has a output.
                if is_dygraph_api(child_node):
                    return
                else:
                    self._visit_Call(child_node)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        # Replace API `to_variable` with `fluid.layers.assign`
        if is_to_variable(node):
            self._update_feed_dict(node)
            node = to_assign_node(node)
            return node

        func_name = astor.to_source(gast.gast_to_ast(node.func))

        if self._is_dygraph_forward(func_name):
            class_node = self._get_class_node(func_name)
            static_node = to_static_ast(node, class_node)
            return static_node
        else:
            return node

    def _is_dygraph_forward(self, func_id):
        return func_id in self.class_node_dict

    def _get_class_node(self, func_id):
        return self.class_node_dict[func_id]

    def _update_class_node_dict(self, node):
        assert isinstance(node, gast.Assign)
        node_value = node.value
        if isinstance(node_value, gast.Call):
            if is_to_variable(node_value):
                return False

            if is_dygraph_api(node_value):
                dygraph_api = node_value.func.attr
                if not dygraph_class_to_static_api.get(dygraph_api):
                    return False

                update_args_of_func(node_value, node_value, "__init__")
                target_str = astor.to_source(gast.gast_to_ast(node.targets[0]))
                self.class_node_dict[target_str] = node_value
                return True
            # TODO: node.value is not dygraph class
        return False

    def _update_feed_dict(self, node):
        assert isinstance(node, gast.Call)

        value_node = None
        for kw in node.keywords:
            if kw.arg == 'value':
                value_node = kw.value  # eg: `a` for "value=a "
        if not value_node:
            value_node = node.args[0]

        if not isinstance(value_node, gast.Name):
            return
        else:
            var_name = value_node.id
            feed_var_name = unique_name.generate(var_name)  # eg: "a_0"
            self.feed_name_to_arg_id[
                feed_var_name] = var_name  # eg: "a_0" : "a"

    def get_feed_name_to_arg_id(self):
        return self.feed_name_to_arg_id
