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

import copy
import gast

from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_api
from paddle.fluid.dygraph.dygraph_to_static.utils import SplitAssignTransformer
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor


def create_convert_shape_node(var_shape_node,
                              slice_node=None,
                              in_control_flow=False):
    assert isinstance(var_shape_node, (gast.Attribute, gast.Subscript))

    if isinstance(var_shape_node, gast.Attribute):
        args = [ast_to_source_code(var_shape_node.value).strip()]
        if slice_node:
            args.append(ast_to_source_code(slice_node).strip())

        convert_var_shape_func = "paddle.jit.dy2static.convert_var_shape({}, in_control_flow={})".format(
            ",".join(args), in_control_flow)

        api_shape_node = gast.parse(convert_var_shape_func).body[0].value
        return api_shape_node

    if isinstance(var_shape_node, gast.Subscript):
        result_node = copy.deepcopy(var_shape_node)
        result_node = create_convert_shape_node(
            result_node.value, result_node.slice, in_control_flow)
        return result_node


class TensorShapeTransformer(gast.NodeTransformer):
    """
    This class transforms variable.shape used in Paddle Apis or control flow conditions into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of TensorShapeTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.name_to_var_shape = {}

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        var_env = self.static_analysis_visitor.get_var_env()
        var_env.cur_scope = var_env.cur_scope.sub_scopes[0]
        self.scope_var_type_dict = var_env.get_scope_var_type()

    def transform(self):
        SplitAssignTransformer(self.root).transform()
        self.visit(self.root)

    def visit_Assign(self, node):
        if self._update_name_to_var_shape(node):
            return node
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        value_node = node.value
        slice_node = node.slice
        if isinstance(value_node, gast.Name):
            if self._is_var_shape(value_node) and self._used_by_paddle_api(
                    value_node):
                var_shape_node = self.name_to_var_shape[value_node.id]
                return create_convert_shape_node(var_shape_node, slice_node)

        if isinstance(value_node, gast.Attribute):
            if self._used_by_paddle_api(value_node) and self._is_var_shape(
                    value_node):
                return create_convert_shape_node(value_node, slice_node)

        return node

    def visit_Attribute(self, node):
        if self._used_by_paddle_api(node):
            if self._is_var_shape(node):
                return create_convert_shape_node(node)
        return node

    def visit_Name(self, node):
        if self._is_var_shape(node):
            if self._used_by_paddle_api(node):
                var_shape_node = self.name_to_var_shape[node.id]
                return create_convert_shape_node(var_shape_node)
        return node

    def visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if is_paddle_api(node):
            # Visit gast.Attribute and gast.Name to replace var.shape if necessary.
            self.generic_visit(node)

        return node

    def visit_If(self, node):
        # Call generic_visit first to transform var.shape that is used in Paddle Api.
        self.generic_visit(node)
        cond = node.test
        self._transform_var_shape_if_necessary(cond)

        return node

    def visit_While(self, node):
        self.generic_visit(node)
        cond = node.test
        self._transform_var_shape_if_necessary(cond)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        iter = node.iter
        self._transform_var_shape_if_necessary(iter)

        # If var.shape is a gast.Name and it is used in range function, transform it
        self._transform_var_shape_in_range(node)
        return node

    def _transform_var_shape_in_range(self, node):
        assert isinstance(node, gast.For)
        if not isinstance(node.iter, gast.Call):
            return False
        if not isinstance(node.iter.func, gast.Name):
            return False
        if node.iter.func.id != "range":
            return False
        args = node.iter.args
        for idx, arg in enumerate(args):
            if isinstance(arg, gast.Name) and self._is_var_shape(arg):
                args[idx] = create_convert_shape_node(self.name_to_var_shape[
                    arg.id])

        return True

    def _transform_var_shape_if_necessary(self, cond):
        need_transformed = False
        for child_node in gast.walk(cond):
            var_shape_node = None
            if isinstance(child_node, (gast.Attribute, gast.Subscript)):
                if self._is_var_shape(child_node):
                    var_shape_node = child_node
            elif isinstance(child_node, (gast.Name)):
                if self._is_var_shape(child_node):
                    var_shape_node = self.name_to_var_shape[child_node.id]

            if var_shape_node:
                need_transformed = True
                wrapper_node = self.node_to_wrapper_map.get(child_node)
                parent_node = wrapper_node.parent.node
                for field, value in gast.iter_fields(parent_node):
                    if child_node is value:
                        setattr(parent_node, field,
                                create_convert_shape_node(var_shape_node, None,
                                                          True))
                        break
                    # Some child_node may be in a list such as gast.Compare
                    if isinstance(value, list):
                        has_converted_shape = False
                        for i, v in enumerate(value):
                            if child_node is v:
                                value[i] = create_convert_shape_node(
                                    var_shape_node, None, True)
                                has_converted_shape = True
                                break
                        if has_converted_shape:
                            break
        return need_transformed

    def _used_by_paddle_api(self, node):
        """
        Whether node is used in paddle api as arguments.
        For example:
            1) Return True in `paddle.relu(x)` where node is `x` (gast.Name)
            2) Return True in `paddle.add(self.x)` where node is `self.x` (gast.Attribute)
            3) Return False in `paddle.add(self.x)` where node is `paddle.add` (gast.Attribute),
               because the role of node is not arguments but `gast.Call.func`.
        """
        assert isinstance(node, (gast.Attribute, gast.Name))
        wrapper_node = self.node_to_wrapper_map.get(node)
        if not wrapper_node:
            # Transformed node is not in node_to_wrapper_map
            return False
        while wrapper_node.parent:
            parent_node = wrapper_node.parent.node
            if isinstance(parent_node, gast.Call):
                # Note(Aurelius84): Filter the case when the role of node is `gast.Call.func`.
                if is_paddle_api(parent_node) and parent_node.func != node:
                    return True
                else:
                    return False
            wrapper_node = wrapper_node.parent

        return False

    def _is_var_shape(self, node):
        """
        Return True if node is like `x.shape` or `x.shape[0]`, return False otherwise.
        """
        if not isinstance(node, (gast.Name, gast.Attribute, gast.Subscript)):
            return False

        if isinstance(node, gast.Name) and node.id in self.name_to_var_shape:
            return True

        if isinstance(node, gast.Attribute):
            if node.attr != 'shape':
                return False

            if not isinstance(node.value, gast.Name):
                return False

            return True

        if isinstance(node, gast.Subscript):
            value_node = node.value
            return self._is_var_shape(value_node)

        return False

    def _update_name_to_var_shape(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        value_node = node.value

        if isinstance(target_node, gast.Tuple):
            has_updated = False
            for idx, element in enumerate(target_node.elts):
                target_id = ast_to_source_code(element).strip()

                if isinstance(value_node, gast.Name):
                    if value_node.id in self.name_to_var_shape:
                        index_value_node = gast.Constant(value=idx, kind=None)
                        slice_index_node = gast.Index(value=index_value_node)
                        var_shape_node = self.name_to_var_shape[value_node.id]
                        sub_node = gast.Subscript(
                            value=var_shape_node,
                            slice=slice_index_node,
                            ctx=gast.Load())
                        self.name_to_var_shape[target_id] = sub_node
                        has_updated = True
                if isinstance(value_node, gast.Attribute):
                    if self._is_var_shape(value_node):  # eg: x.shape
                        index_value_node = gast.Constant(value=idx, kind=None)
                        slice_index_node = gast.Index(value=index_value_node)
                        sub_node = gast.Subscript(
                            value=value_node,
                            slice=slice_index_node,
                            ctx=gast.Load())
                        self.name_to_var_shape[target_id] = sub_node
                        has_updated = True

            return has_updated
        else:
            target_id = ast_to_source_code(target_node).strip()

            if isinstance(value_node, gast.Name):
                if self._is_var_shape(value_node):
                    self.name_to_var_shape[target_id] = self.name_to_var_shape[
                        value_node.id]
                    return True
            if isinstance(value_node, gast.Attribute):
                if self._is_var_shape(value_node):  # eg: x.shape
                    self.name_to_var_shape[target_id] = value_node
                    return True
            if isinstance(value_node, gast.Subscript):
                if isinstance(value_node.value, gast.Attribute):
                    if self._is_var_shape(value_node.value):  # eg: x.shape[0]
                        self.name_to_var_shape[target_id] = value_node
                        return True
        return False
