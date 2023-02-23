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

from paddle.fluid import unique_name
from paddle.utils import gast

from .base_transformer import BaseTransformer
from .break_continue_transformer import ForToWhileTransformer
from .utils import (
    ORIGI_INFO,
    Dygraph2StaticException,
    ast_to_source_code,
    index_in_list,
)

__all__ = []

# Constant for the name of the variable which stores the boolean state that we
# should return
RETURN_PREFIX = '__return'

# Constant for the name of the variable which stores the final return value
RETURN_VALUE_PREFIX = '__return_value'

# Constant for the name of variables to initialize the __return_value
RETURN_VALUE_INIT_NAME = '__return_value_init'

# Constant magic number representing returning no value. This constant amis to
# support returning various lengths of variables. Static graph must have fixed
# size of fetched output while dygraph can have flexible lengths of output, to
# solve it in dy2stat, we put float64 value with this magic number at Static
# graph as a place holder to indicate the returning placeholder means no value
# should return.

# Assign not support float64, use float32 value as magic number.
RETURN_NO_VALUE_MAGIC_NUM = 1.77113e27
RETURN_NO_VALUE_VAR_NAME = "__no_value_return_var"


def get_return_size(return_node):
    assert isinstance(return_node, gast.Return), "Input is not gast.Return node"
    return_length = 0
    if return_node.value is not None:
        if isinstance(return_node.value, gast.Tuple):
            return_length = len(return_node.value.elts)
        else:
            return_length = 1
    return return_length


class ReplaceReturnNoneTransformer(BaseTransformer):
    """
    Replace 'return None' to  'return' because 'None' cannot be a valid input
    in control flow. In ReturnTransformer single 'Return' will be appended no
    value placeholder
    """

    def __init__(self, root_node):
        self.root = root_node

    def transform(self):
        self.visit(self.root)

    def visit_Return(self, node):
        if isinstance(node.value, gast.Name) and node.value.id == 'None':
            node.value = None
            return node
        if isinstance(node.value, gast.Constant) and node.value.value is None:
            node.value = None
            return node
        return node


class ReturnAnalysisVisitor(gast.NodeVisitor):
    """
    Visits gast Tree and analyze the information about 'return'.
    """

    def __init__(self, root_node):
        self.root = root_node
        assert isinstance(
            self.root, gast.FunctionDef
        ), "Input is not gast.FunctionDef node"

        # the number of return statements
        self.count_return = 0

        # maximum number of variables
        self.max_return_length = 0

        self.visit(self.root)

    def visit_FunctionDef(self, node):
        """
        don't analysis closure, just analyze current func def level.
        """
        if node == self.root:
            self.generic_visit(node)

    def visit_Return(self, node):
        self.count_return += 1

        return_length = get_return_size(node)
        self.max_return_length = max(self.max_return_length, return_length)

        self.generic_visit(node)

    def get_func_return_count(self):
        return self.count_return

    def get_func_max_return_length(self):
        return self.max_return_length


class ReturnTransformer(BaseTransformer):
    """
    Transforms return statements into equivalent python statements containing
    only one return statement at last. The basics idea is using a return value
    variable to store the early return statements and boolean states with
    if-else to skip the statements after the return.

    Go through all the function definition and call SingleReturnTransformer for each function.
    SingleReturnTransformer don't care the nested function def.
    """

    def __init__(self, wrapper_root):
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        pre_transformer = ReplaceReturnNoneTransformer(self.root)
        pre_transformer.transform()

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        node = SingleReturnTransformer(node).transform()
        return node


class SingleReturnTransformer(BaseTransformer):
    """
    This function only apply to single function. don't care the nested function_def
    """

    def __init__(self, root):
        self.root = root
        assert isinstance(
            self.root, gast.FunctionDef
        ), "Input is not gast.FunctionDef node"

        self.ancestor_nodes = []

        # The name of return placeholder
        self.return_value_name = None

        # Every return stmt corresponds to a bool value variable, and return name is the name of the boolean variable
        self.return_name = []

        self.pre_analysis = None

    def assert_parent_is_not_while(self, parent_node_of_return):
        if isinstance(parent_node_of_return, (gast.While, gast.For)):
            raise Dygraph2StaticException(
                "Found return statement in While or For body and loop "
                "is meaningless, please check you code and remove return in while/for."
            )

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
        ret = super().visit(node)
        self.ancestor_nodes.pop()
        return ret

    def visit_FunctionDef(self, node):
        """
        don't analysis closure, just analyze current func def level.
        """
        if node == self.root:
            self.generic_visit(node)
        return node

    def append_assign_to_return_node(
        self, value, parent_node_of_return, return_name, assign_nodes
    ):
        self.assert_parent_is_not_while(parent_node_of_return)
        assert value in [True, False], "value must be True or False."
        if isinstance(parent_node_of_return, gast.If):
            # Prepend control flow boolean nodes such as '__return@1 = True'
            node_str = "{} = _jst.create_bool_as_type({}, {})".format(
                return_name,
                ast_to_source_code(parent_node_of_return.test).strip(),
                value,
            )

            assign_node = gast.parse(node_str).body[0]
            assign_nodes.append(assign_node)

    def transform(self):
        node = self.root
        self.pre_analysis = ReturnAnalysisVisitor(node)
        max_return_length = self.pre_analysis.get_func_max_return_length()
        while self.pre_analysis.get_func_return_count() > 0:
            # every visit will decrease the number of returns.
            # so we need a while.
            self.visit(node)
            self.pre_analysis = ReturnAnalysisVisitor(node)

        if max_return_length == 0:
            return node

        # Prepend initialization of final return and append final return statement
        value_name = self.return_value_name
        if value_name is not None:
            node.body.append(
                gast.Return(
                    value=gast.Name(
                        id=value_name,
                        ctx=gast.Load(),
                        annotation=None,
                        type_comment=None,
                    )
                )
            )
            assign_return_value_node = gast.Assign(
                targets=[
                    gast.Name(
                        id=value_name,
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None,
                    )
                ],
                value=gast.Constant(kind=None, value=None),
            )
            node.body.insert(0, assign_return_value_node)

        # Prepend no value placeholders
        return node

    def visit_Return(self, node):
        return_name = unique_name.generate(RETURN_PREFIX)
        self.return_name.append(return_name)
        max_return_length = self.pre_analysis.get_func_max_return_length()
        parent_node_of_return = self.ancestor_nodes[-2]

        for ancestor_index in reversed(range(len(self.ancestor_nodes) - 1)):
            ancestor = self.ancestor_nodes[ancestor_index]
            cur_node = self.ancestor_nodes[ancestor_index + 1]

            def _deal_branches(branch_name):
                if hasattr(ancestor, branch_name):
                    branch_node = getattr(ancestor, branch_name)
                    if index_in_list(branch_node, cur_node) != -1:
                        if cur_node == node:
                            self._replace_return_in_stmt_list(
                                branch_node,
                                cur_node,
                                return_name,
                                max_return_length,
                                parent_node_of_return,
                            )
                        self._replace_after_node_to_if_in_stmt_list(
                            branch_node,
                            cur_node,
                            return_name,
                            parent_node_of_return,
                        )

            _deal_branches("body")
            _deal_branches("orelse")
            # If return node in while loop, add `not return_name` in gast.While.test
            if isinstance(ancestor, gast.While):
                cond_var_node = gast.UnaryOp(
                    op=gast.Not(),
                    operand=gast.Name(
                        id=return_name,
                        ctx=gast.Load(),
                        annotation=None,
                        type_comment=None,
                    ),
                )
                ancestor.test = gast.BoolOp(
                    op=gast.And(), values=[ancestor.test, cond_var_node]
                )
                continue

            # If return node in for loop, add `not return_name` in gast.While.test
            if isinstance(ancestor, gast.For):
                cond_var_node = gast.UnaryOp(
                    op=gast.Not(),
                    operand=gast.Name(
                        id=return_name,
                        ctx=gast.Load(),
                        annotation=None,
                        type_comment=None,
                    ),
                )
                parent_node = self.ancestor_nodes[ancestor_index - 1]
                for_to_while = ForToWhileTransformer(
                    parent_node, ancestor, cond_var_node
                )
                new_stmts = for_to_while.transform()
                while_node = new_stmts[-1]
                self.ancestor_nodes[ancestor_index] = while_node

            if ancestor == self.root:
                break
        # return_node is replaced so we shouldn't return here

    def _replace_return_in_stmt_list(
        self,
        stmt_list,
        return_node,
        return_name,
        max_return_length,
        parent_node_of_return,
    ):

        assert max_return_length >= 0, "Input illegal max_return_length"
        i = index_in_list(stmt_list, return_node)
        if i == -1:
            return False

        assign_nodes = []
        self.append_assign_to_return_node(
            True, parent_node_of_return, return_name, assign_nodes
        )

        return_length = get_return_size(return_node)
        # In this case we should NOT append RETURN_NO_VALUE placeholder
        if return_node.value is not None:
            if self.return_value_name is None:
                self.return_value_name = unique_name.generate(
                    RETURN_VALUE_PREFIX
                )

            assign_nodes.append(
                gast.Assign(
                    targets=[
                        gast.Name(
                            id=self.return_value_name,
                            ctx=gast.Store(),
                            annotation=None,
                            type_comment=None,
                        )
                    ],
                    value=return_node.value,
                )
            )
            return_origin_info = getattr(return_node, ORIGI_INFO, None)
            setattr(assign_nodes[-1], ORIGI_INFO, return_origin_info)

        # If there is a return in the body or else of if, the remaining statements
        # will not be executed, so they can be properly replaced.
        stmt_list[i:] = assign_nodes
        return True

    def _replace_after_node_to_if_in_stmt_list(
        self, stmt_list, node, return_name, parent_node_of_return
    ):
        i = index_in_list(stmt_list, node)
        if i < 0 or i >= len(stmt_list):
            return False
        if i == len(stmt_list) - 1:
            # No need to add, we consider this as added successfully
            return True

        if_stmt = gast.If(
            test=gast.UnaryOp(
                op=gast.Not(),
                operand=gast.Name(
                    id=return_name,
                    ctx=gast.Store(),
                    annotation=None,
                    type_comment=None,
                ),
            ),
            body=stmt_list[i + 1 :],
            orelse=[],
        )

        stmt_list[i + 1 :] = [if_stmt]

        # Here assume that the parent node of return is gast.If
        assign_nodes = []
        self.append_assign_to_return_node(
            False, parent_node_of_return, return_name, assign_nodes
        )
        stmt_list[i:i] = assign_nodes
        return True
