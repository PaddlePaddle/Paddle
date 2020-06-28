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

__all__ = [
    'RETURN_NO_VALUE_MAGIC_NUM', 'RETURN_NO_VALUE_VAR_NAME', 'ReturnTransformer'
]

# Constant for the name of the variable which stores the boolean state that we
# should return
RETURN_PREFIX = '__return'

# Constant for the name of the variable which stores the final return value
RETURN_VALUE_PREFIX = '__return_value'

# Constant magic number representing returning no value. This constant amis to
# support returning various lengths of variables. Static graph must have fixed
# size of fetched output while dygraph can have flexible lengths of output, to
# solve it in dy2stat, we put float64 value with this magic number at Static
# graph as a place holder to indicate the returning placeholder means no value
# should return.
RETURN_NO_VALUE_MAGIC_NUM = 1.771137e+279
RETURN_NO_VALUE_VAR_NAME = str(RETURN_NO_VALUE_MAGIC_NUM)


def get_return_size(return_node):
    assert isinstance(return_node, gast.Return), "Input is not gast.Return node"
    return_length = 0
    if return_node.value is not None:
        if isinstance(return_node.value, gast.Tuple):
            return_length = len(return_node.value.elts)
        else:
            return_length = 1
    return return_length


class ReturnAnalysisVisitor(gast.NodeVisitor):
    """
    Visits gast Tree and analyze the information about 'return'.
    """

    def __init__(self, root_node):
        self.root = root_node

        # A list to store where the current function is.
        self.function_def = []

        # Mapping from gast.FunctionDef node to the number of return statements
        # Python allows define function inside function so we have to handle it
        self.count_return = {}

        # Mapping from gast.FunctionDef node to the maximum number of variables
        # returned by the function's return statement
        self.max_return_length = {}
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

        return_length = get_return_size(node)
        if cur_func in self.max_return_length:
            self.max_return_length[cur_func] = max(
                self.max_return_length[cur_func], return_length)
        else:
            self.max_return_length[cur_func] = return_length

        self.generic_visit(node)

    def get_func_return_count(self, func_node):
        return self.count_return[func_node]

    def get_func_max_return_length(self, func_node):
        return self.max_return_length[func_node]


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

        self.pre_analysis = ReturnAnalysisVisitor(node)

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

        self.pre_analysis = ReturnAnalysisVisitor(node)
        while self.pre_analysis.get_func_return_count(node) > 1:
            self.generic_visit(node)
            self.pre_analysis = ReturnAnalysisVisitor(node)

        # prepend initialization of final return and append final return statement
        value_name = self.return_value_name[node]
        if value_name is not None:
            node.body.append(
                gast.Return(value=gast.Name(
                    id=value_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None)))
            assign_zero_node = create_fill_constant_node(value_name, 0.0)
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
        max_return_length = self.pre_analysis.get_max_return_length(
            cur_func_node)
        for ancestor_index in reversed(range(len(self.ancestor_nodes) - 1)):
            ancestor = self.ancestor_nodes[ancestor_index]
            cur_node = self.ancestor_nodes[ancestor_index + 1]
            if hasattr(ancestor,
                       "body") and index_in_list(ancestor.body, cur_node) != -1:
                if cur_node == node:
                    self._replace_return_in_stmt_list(ancestor.body, cur_node,
                                                      return_name)
                self._replace_after_node_to_if_in_stmt_list(
                    ancestor.body, cur_node, return_name, max_return_length)
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

    def _replace_return_in_stmt_list(self, stmt_list, return_node, return_name,
                                     max_return_length):
        assert max_return_length >= 0, "Input illegal max_return_length"
        i = index_in_list(stmt_list, return_node)
        if i == -1:
            return False
        assign_nodes = [create_fill_constant_node(return_name, True)]

        return_length = get_return_size(return_node)
        if return_length < max_return_length:
            # In this case we should append RETURN_NO_VALUE placeholder
            #
            # max_return_length must be >= 1 here because return_length will be
            # 0 at least.
            if self.return_value_name[cur_func_node] is None:
                self.return_value_name[cur_func_node] = unique_name.generate(
                    RETURN_VALUE_PREFIX)
            placeholder_node = create_fill_constant_node(
                RETURN_NO_VALUE_VAR_NAME, RETURN_NO_VALUE_MAGIC_NUM)

            # Handle tuple/non-tuple case
            if max_return_length == 1:
                assign_nodes.append(
                    gast.Assign(
                        targets=[
                            gast.Name(
                                id=self.return_value_name[cur_func_node],
                                ctx=gast.Store(),
                                annotation=None,
                                type_comment=None)
                        ],
                        value=placeholder_node))
            else:
                # max_return_length > 1 which means we should assign tuple
                fill_tuple = [
                    create_fill_constant_node(RETURN_NO_VALUE_VAR_NAME,
                                              RETURN_NO_VALUE_MAGIC_NUM)
                    for i in range(max_return_length - return_length)
                ]
                if return_node.value is not None:
                    if isinstance(return_node.value, gast.Tuple):
                        fill_tuple[:0] = return_node.value.elts
                    else:
                        fill_tuple.insert(0, return_node.value)

                assign_nodes.append(
                    gast.Assign(
                        targets=[
                            gast.Name(
                                id=self.return_value_name[cur_func_node],
                                ctx=gast.Store(),
                                annotation=None,
                                type_comment=None)
                        ],
                        value=gast.Tuple(
                            elts=fill_tuple, expr_context=gast.Load())))
        else:
            # In this case we should NOT append RETURN_NO_VALUE placeholder
            if return_node.value is not None:
                cur_func_node = self.function_def[-1]
                if self.return_value_name[cur_func_node] is None:
                    self.return_value_name[
                        cur_func_node] = unique_name.generate(
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
