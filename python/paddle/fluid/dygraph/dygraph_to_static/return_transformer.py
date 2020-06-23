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

from paddle.fluid import unique_name
from paddle.fluid.dygraph.dygraph_to_static.utils import index_in_list
from paddle.fluid.dygraph.dygraph_to_static.break_continue_transformer import ForToWhileTransformer
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import create_fill_constant_node

__all__ = ['ReturnTransformer']

# Constant for the name of the variable which stores the boolean state that we
# should return
RETURN_PREFIX = '__return'

# Constant for the name of the variable which stores the final return value
RETURN_VALUE_PREFIX = '__return_value'


class ReturnPreAnalysisVisitor(gast.NodeVisitor):
    """
    Visits gast Tree and pre-analyze the information about 'return'.
    """

    def __init__(self, root_node):
        self.root = root_node

        # A list to store where the current function is.
        self.function_def = []

        # Mapping from gast.FunctionDef node to the number of return statements
        # Python allows define function inside function so we have to handle it
        self.count_return = {}
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        self.function_def.append(node)
        self.count_return[node] = 0
        self.generic_visit(node)
        self.function_def.pop()
        return node

    def visit_Return(self, node):
        assert len(
            self.function_def) > 0, "Found 'return' statement out of function."
        cur_func = self.function_def[-1]
        if cur_func in self.count_return:
            self.count_return[cur_func] += 1
        else:
            self.count_return[cur_func] = 1
        self.generic_visit(node)

    def get_func_return_count(self, func_node):
        return self.count_return[func_node]

    def set_func_return_count(self, func_node, count):
        self.count_return[func_node] = count


class ReturnTransformer(gast.NodeTransformer):
    """
    Transforms return statements into equivalent python statements containing
    only one return statement at last. The basics idea is using a return value
    variable to store the early return statements and boolean states with
    if-else to skip the statements after the return.

    """

    def __init__(self, wrapper_root):
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.ancestor_nodes = []

        # The name of the variable which stores the final return value
        # Mapping from FunctionDef node to string
        self.return_value_name = {}
        # The names of the variable which stores the boolean state that skip
        # statments. Mapping from FunctionDef node to list
        self.return_name = {}
        # A list of FunctionDef to store where the current function is.
        self.function_def = []

    def transform(self):
        self.visit(self.root)

    def generic_visit(self, node):
        # Because we change ancestor nodes during visit_Return, not current
        # node, original generic_visit of NodeTransformer will visit node
        # which may be deleted. To prevent that node being added into
        # transformed AST, We self-write a generic_visit and visit
        for field, value in gast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, gast.AST):
                        self.visit(item)
            elif isinstance(value, gast.AST):
                self.visit(value)

    def visit(self, node):
        """
        Self-defined visit for appending ancestor
        """
        self.ancestor_nodes.append(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()
        return ret

    def visit_FunctionDef(self, node):
        self.function_def.append(node)
        self.return_value_name[node] = None
        self.return_name[node] = []

        pre_analysis = ReturnPreAnalysisVisitor(node)
        while pre_analysis.get_func_return_count(node) > 1:
            self.generic_visit(node)
            pre_analysis = ReturnPreAnalysisVisitor(node)

        # prepend initialization of final return and append final return statement
        value_name = self.return_value_name[node]
        if value_name is not None:
            node.body.append(
                gast.Return(value=gast.Name(
                    id=value_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None)))
            assign_zero_node = create_fill_constant_node(value_name, 0)
            node.body.insert(0, assign_zero_node)
        # Prepend control flow boolean nodes such as '__return@1 = False'
        for name in self.return_name[node]:
            assign_false_node = create_fill_constant_node(name, False)
            node.body.insert(0, assign_false_node)

        self.function_def.pop()
        return node

    def visit_Return(self, node):
        cur_func_node = self.function_def[-1]
        return_name = unique_name.generate(RETURN_PREFIX)
        self.return_name[cur_func_node].append(return_name)
        for ancestor_index in reversed(range(len(self.ancestor_nodes) - 1)):
            ancestor = self.ancestor_nodes[ancestor_index]
            cur_node = self.ancestor_nodes[ancestor_index + 1]
            if hasattr(ancestor,
                       "body") and index_in_list(ancestor.body, cur_node) != -1:
                if cur_node == node:
                    self._replace_return_in_stmt_list(ancestor.body, cur_node,
                                                      return_name)
                self._replace_after_node_to_if_in_stmt_list(
                    ancestor.body, cur_node, return_name)
            elif hasattr(ancestor, "orelse") and index_in_list(ancestor.orelse,
                                                               cur_node) != -1:
                if cur_node == node:
                    self._replace_return_in_stmt_list(ancestor.orelse, cur_node,
                                                      return_name)
                self._replace_after_node_to_if_in_stmt_list(
                    ancestor.orelse, cur_node, return_name)

            if isinstance(ancestor, gast.While):
                cond_var_node = gast.UnaryOp(
                    op=gast.Not(),
                    operand=gast.Name(
                        id=return_name,
                        ctx=gast.Load(),
                        annotation=None,
                        type_comment=None))
                ancestor.test = gast.BoolOp(
                    op=gast.And(), values=[ancestor.test, cond_var_node])
                continue

            if isinstance(ancestor, gast.For):
                cond_var_node = gast.UnaryOp(
                    op=gast.Not(),
                    operand=gast.Name(
                        id=return_name,
                        ctx=gast.Load(),
                        annotation=None,
                        type_comment=None))
                parent_node = self.ancestor_nodes[ancestor_index - 1]
                for_to_while = ForToWhileTransformer(parent_node, ancestor,
                                                     cond_var_node)
                new_stmts = for_to_while.transform()
                while_node = new_stmts[-1]
                self.ancestor_nodes[ancestor_index] = while_node

            if ancestor == cur_func_node:
                break
        # return_node is replaced so we shouldn't return here

    def _replace_return_in_stmt_list(self, stmt_list, return_node, return_name):
        i = index_in_list(stmt_list, return_node)
        if i == -1:
            return False
        assign_nodes = [create_fill_constant_node(return_name, True)]
        if return_node.value is not None:
            cur_func_node = self.function_def[-1]
            if self.return_value_name[cur_func_node] is None:
                self.return_value_name[cur_func_node] = unique_name.generate(
                    RETURN_VALUE_PREFIX)
            assign_nodes.append(
                gast.Assign(
                    targets=[
                        gast.Name(
                            id=self.return_value_name[cur_func_node],
                            ctx=gast.Store(),
                            annotation=None,
                            type_comment=None)
                    ],
                    value=return_node.value))
        stmt_list[i:] = assign_nodes
        return True

    def _replace_after_node_to_if_in_stmt_list(self, stmt_list, node,
                                               return_name):
        i = index_in_list(stmt_list, node)
        if i < 0 or i >= len(stmt_list):
            return False
        if i == len(stmt_list) - 1:
            # No need to add, we consider this as added successfully
            return True
        if_stmt = gast.If(test=gast.UnaryOp(
            op=gast.Not(),
            operand=gast.Name(
                id=return_name,
                ctx=gast.Store(),
                annotation=None,
                type_comment=None)),
                          body=stmt_list[i + 1:],
                          orelse=[])
        stmt_list[i + 1:] = [if_stmt]
        return True
