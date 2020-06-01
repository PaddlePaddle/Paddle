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
from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_api
from paddle.fluid.dygraph.dygraph_to_static.utils import create_api_shape_node
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, NodeVarType
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor


class TensorShapeTransformer(gast.NodeTransformer):
    """
    This class transforms Tensor.shape used in Paddle Apis and control flow conditions into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of TensorShapeTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.name_to_tensor_shape = {}

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        var_env = self.static_analysis_visitor.get_var_env()
        var_env.cur_scope = var_env.cur_scope.sub_scopes[0]
        self.scope_var_type_dict = var_env.get_scope_var_type()

    def transform(self):
        self.visit(self.root)

    def visit_Assign(self, node):
        if self._update_name_to_tensor_shape(node):
            return node
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        if self._used_by_paddle_api(node):
            if self.is_tensor_shape(node):
                return create_api_shape_node(node)
        return node

    def visit_Name(self, node):
        if node.id in self.name_to_tensor_shape:
            if self._used_by_paddle_api(node):
                tensor_shape_node = self.name_to_tensor_shape[node.id]
                return create_api_shape_node(tensor_shape_node)
        return node

    def visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if is_paddle_api(node):
            # Visit gast.Attribute and gast.Name to replace tensor.shape if necessary.
            self.generic_visit(node)

        return node

    def visit_If(self, node):
        # Call generic_visit first to transform Tensor.shape that is used in Paddle Api.
        self.generic_visit(node)
        cond = node.test
        self._transform_tensor_shape_if_necessary(cond)
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        cond = node.test
        self._transform_tensor_shape_if_necessary(cond)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        iter = node.iter
        self._transform_tensor_shape_if_necessary(iter)

        # If tensor.shape is a gast.Name and it is used in range function, transform it
        self._transform_tensor_shape_in_range(node)
        return node

    def _transform_tensor_shape_in_range(self, node):
        assert isinstance(node, gast.For)
        if not isinstance(node.iter, gast.Call):
            return False
        if not isinstance(node.iter.func, gast.Name):
            return False
        if node.iter.func.id != "range":
            return False
        args = node.iter.args
        for idx, arg in enumerate(args):
            if isinstance(arg,
                          gast.Name) and arg.id in self.name_to_tensor_shape:
                args[idx] = create_api_shape_node(self.name_to_tensor_shape[
                    arg.id])

        return True

    def _transform_tensor_shape_if_necessary(self, cond):
        for child_node in gast.walk(cond):
            tensor_shape_node = None
            if isinstance(child_node, (gast.Attribute)):
                if self.is_tensor_shape(child_node):
                    tensor_shape_node = child_node
            elif isinstance(child_node, (gast.Name)):
                if child_node.id in self.name_to_tensor_shape:
                    tensor_shape_node = self.name_to_tensor_shape[child_node.id]

            if tensor_shape_node:
                wrapper_node = self.node_to_wrapper_map.get(child_node)
                parent_node = wrapper_node.parent.node
                for field, value in gast.iter_fields(parent_node):
                    if child_node is value:
                        setattr(parent_node, field,
                                create_api_shape_node(tensor_shape_node))
                        break

    def _used_by_paddle_api(self, node):
        assert isinstance(node, (gast.Attribute, gast.Name))
        wrapper_node = self.node_to_wrapper_map.get(node)
        if not wrapper_node:
            # Transformed node is not in node_to_wrapper_map
            return False
        while wrapper_node.parent:
            parent_node = wrapper_node.parent.node
            if isinstance(parent_node, gast.Call):
                if is_paddle_api(parent_node):
                    return True
                else:
                    return False
            wrapper_node = wrapper_node.parent

        return False

    def is_tensor_shape(self, node):
        """
        Return True if node is like `x.shape` and x is Tensor, return False otherwise.
        """
        assert isinstance(node, gast.Attribute)
        if node.attr != 'shape':
            return False

        try:
            value_id = node.value.id
        except AttributeError:
            return False

        if value_id in self.name_to_tensor_shape:
            return True

        # TODO: `value_id` may be not in scope_var_type_dict if `value_id` is the arg of decorated function
        # Need a better way to confirm whether `value_id` is a Tensor.
        try:
            var_type_set = self.scope_var_type_dict[value_id]
        except KeyError:
            return False

        if NodeVarType.NUMPY_NDARRAY in var_type_set:
            return False
        if NodeVarType.TENSOR not in var_type_set and NodeVarType.PADDLE_RETURN_TYPES not in var_type_set:
            return False

        return True

    def _update_name_to_tensor_shape(self, node):
        assert isinstance(node, gast.Assign)
        # TODO: Consider node has more than one target. eg: x, y = a, Tensor.shape[1]
        target_node = node.targets[0]
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        value_node = node.value

        if isinstance(value_node, gast.Name):
            if value_node.id in self.name_to_tensor_shape:
                self.name_to_tensor_shape[
                    target_id] = self.name_to_tensor_shape[value_node.id]
                return True
        if isinstance(value_node, gast.Attribute):
            if self.is_tensor_shape(value_node):  # eg: x.shape
                self.name_to_tensor_shape[target_id] = value_node
                return True
        if isinstance(value_node, gast.Subscript):
            if isinstance(value_node.value, gast.Attribute):
                if self.is_tensor_shape(value_node.value):  # eg: x.shape[0]
                    self.name_to_tensor_shape[target_id] = value_node
                    return True
        return False
