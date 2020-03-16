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

import gast
import astor
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, NodeVarType, StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.utils import is_control_flow_to_transform


class ListTransformer(gast.NodeTransformer):
    """
    This class transforms python list used in control flow into Static Graph Ast
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of ListTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.name_of_list_set = set()
        self.list_name_to_updated = dict()

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        var_env = self.static_analysis_visitor.get_var_env()
        var_env.cur_scope = var_env.cur_scope.sub_scopes[0]
        self.scope_var_type_dict = var_env.get_scope_var_type()

    def transform(self):
        self.visit(self.root)
        self.replace_list_with_tensor_array(self.root)

    def visit_Assign(self, node):
        self._update_list_name_to_updated(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def replace_list_with_tensor_array(self, node):
        for child_node in gast.walk(node):
            if isinstance(child_node, gast.Assign):
                if self._need_to_create_tensor_array(child_node):
                    child_node.value = self._create_tensor_array()

    def _transform_list_append_in_control_flow(self, node):
        for child_node in gast.walk(node):
            if self._need_to_array_write_node(child_node):
                child_node.value = \
                    self._to_array_write_node(child_node.value)

    def _need_to_array_write_node(self, node):
        if isinstance(node, gast.Expr):
            if isinstance(node.value, gast.Call):
                if self._is_list_append_tensor(node.value):
                    return True

        return False

    def _is_list_append_tensor(self, node):
        """
        a.append(b): a is list, b is Tensor
        self.x.append(b): self.x is list, b is Tensor
        """
        assert isinstance(node, gast.Call)
        # 1. The func is `append`.
        if not isinstance(node.func, gast.Attribute):
            return False
        if node.func.attr != 'append':
            return False

        # 2. It's a `python list` to call append().
        value_name = astor.to_source(gast.gast_to_ast(node.func.value)).strip()
        if value_name not in self.list_name_to_updated:
            return False

        # 3. The arg of append() is one `Tensor`
        # Only one argument is supported in Python list.append()
        if len(node.args) != 1:
            return False
        arg = node.args[0]
        if isinstance(arg, gast.Name):
            # TODO: `arg.id` may be not in scope_var_type_dict if `arg.id` is the arg of decorated function
            # Need a better way to confirm whether `arg.id` is a Tensor.
            try:
                var_type_set = self.scope_var_type_dict[arg.id]
            except KeyError:
                return False

            if NodeVarType.NUMPY_NDARRAY in var_type_set:
                return False
            if NodeVarType.TENSOR not in var_type_set and NodeVarType.PADDLE_RETURN_TYPES not in var_type_set:
                return False
        # else:
        # Todo: Consider that `arg` may be a gast.Call about Paddle Api.
        # eg: list_a.append(fluid.layers.reshape(x))
        # return True
        self.list_name_to_updated[value_name.strip()] = True
        return True

    def _need_to_create_tensor_array(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        if self.list_name_to_updated.get(target_id):
            return True
        return False

    def _create_tensor_array(self):
        # Although `dtype='float32'`, other types such as `int32` can also be supported
        func_code = "fluid.layers.create_array(dtype='float32')"
        func_node = gast.parse(func_code).body[0].value
        return func_node

    def _to_array_write_node(self, node):
        assert isinstance(node, gast.Call)
        array = astor.to_source(gast.gast_to_ast(node.func.value))
        x = astor.to_source(gast.gast_to_ast(node.args[0]))
        i = "fluid.layers.array_length({})".format(array)
        func_code = "fluid.layers.array_write(x={}, i={}, array={})".format(
            x, i, array)
        return gast.parse(func_code).body[0].value

    def _update_list_name_to_updated(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        # TODO: Consider node has more than one target. eg: x, y = a, []
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        value_node = node.value
        if isinstance(value_node, gast.List):
            self.list_name_to_updated[target_id] = False
            return True
        elif target_id in self.name_of_list_set:
            del self.list_name_to_updated[target_id]
        return False
