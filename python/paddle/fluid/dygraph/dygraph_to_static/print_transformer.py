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
import logging

from paddle.fluid import log_helper
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, NodeVarType, StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code

_logger = log_helper.get_logger(
    __name__, logging.WARNING, fmt='%(asctime)s-%(levelname)s: %(message)s')


class PrintTransformer(gast.NodeTransformer):
    """
    This class transforms python print function to fluid.layers.Print.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of PrintTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )

    def transform(self):
        self.visit(self.root)

    # NOTE: deal with print in PY3
    def visit_Call(self, node):
        if isinstance(node.func, gast.Name) and node.func.id == 'print':
            parent_node = self.node_to_wrapper_map[node].parent.node
            if isinstance(parent_node, gast.Expr):
                # NOTE: why need transform to gast.Assign node
                # only fluid.layers.Print(x) will be pruned when exe.run(use_prune=True)
                print_assign_node = self._create_assign_node(node)
                if print_assign_node is not None:
                    return print_assign_node
            else:
                return self._transform_call_node(node)
        return node

    # NOTE: deal with print in PY2
    def visit_Print(self, node):
        print_assign_node = self._create_assign_node(node)
        if print_assign_node is not None:
            return print_assign_node
        return node

    def _transform_call_node(self, node):
        assert isinstance(node, gast.Call), "visit Node is not gast.Call node."
        var_node = self._get_print_var_node(node)
        if var_node is None:
            return node
        if self._need_transform(var_node, node):
            return self._build_print_call_node(var_node)
        return node

    def _create_assign_node(self, node):
        var_node = self._get_print_var_node(node)
        if var_node is None:
            return None
        if self._need_transform(var_node, node):
            return gast.Assign(
                targets=[var_node], value=self._build_print_call_node(var_node))
        return None

    def _build_print_call_node(self, node):
        return gast.Call(
            func=gast.parse('fluid.layers.Print').body[0].value,
            args=[node],
            keywords=[
                gast.keyword(
                    arg='summarize',
                    value=gast.UnaryOp(
                        op=gast.USub(),
                        operand=gast.Constant(
                            value=1, kind=None))), gast.keyword(
                                arg='print_phase',
                                value=gast.Constant(
                                    value='forward', kind=None))
            ])

    def _get_print_var_node(self, node):
        if isinstance(node, gast.Call):
            var_list = node.args
        elif isinstance(node, gast.Print):
            var_list = node.values
            if isinstance(var_list[0], gast.Tuple):
                var_list = var_list[0].elts
        # TODO: support print multiple Var
        if len(var_list) == 1:
            return var_list[0]
        else:
            _logger.warning(
                "ProgramTranslator could not transform printing multiple values like < %s > now and will run it as-is."
                % ast_to_source_code(node).strip())
        return None

    def _need_transform(self, var_node, print_node):
        if isinstance(var_node, gast.Name):
            if self._is_tensor_node(var_node):
                return True
            else:
                _logger.warning(
                    "ProgramTranslator could not transform printing value that are not Tensor like < %s > now and will run it as-is."
                    % ast_to_source_code(print_node).strip())
        else:
            _logger.warning(
                "ProgramTranslator could not transform < %s > now and will run it as-is."
                % ast_to_source_code(print_node).strip())
        return False

    def _is_tensor_node(self, node):
        tensor_types = {NodeVarType.TENSOR, NodeVarType.PADDLE_RETURN_TYPES}
        wrapper_node = self.node_to_wrapper_map.get(node, None)
        if wrapper_node is not None:
            if wrapper_node.node_var_type & tensor_types:
                return True
        return False
