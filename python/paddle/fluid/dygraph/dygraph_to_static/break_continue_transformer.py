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
from paddle.fluid.dygraph.dygraph_to_static.for_to_while_transformer import ForToWhileTransformer

__all__ = ['BreakContinueTransformer']

BREAK_NAME_PREFIX = '__break__'
CONTINUE_NAME_PREFIX = '__continue__'


class BreakContinueTransformer(gast.NodeTransformer):
    """
    Rewrite 'break' and 'continue' key words in a if-else python way to make
    it equivalent to original control flow
    
    The main idea of this class is:

        1. Map the 'break/continue' stmt with an unique boolean variable V.

        2. Find the first ancestor block containing this 'break/continue', a
        block can be a node containing stmt list. We should remove all stmts
        after the 'break/continue' and set the V to True here.

        3. Add 'if V' for stmts in ancestor blocks between the first one
        (exclusive) and the ancestor loop (inclusive)

        4. For 'break' add break into condition of the loop. For 'continue',
        set continue to False at the beginning of each loop

        TODO: more details should be summarized as design document
    """

    def __init__(self, wrapper_root):
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

        self.ancestor_nodes = []

    def transform(self):
        self.visit(self.root)

    def visit(self, node):
        self.ancestor_nodes.append(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor.pop()
        return ret

    def visit_Break(self, node):
        loop_node_index = self._find_ancestor_loop_index(node)
        assert loop_node_index != -1, "SyntaxError: 'break' outside loop"
        loop_node = self.ancestor_nodes[loop_node_index]

        if not isinstance(loop_node, gast.While):
            # "For in range loop" will be transformed into while at
            # ForToWhileTransformer, other "For loop" is unsupported now
            # so we just do nothing and return
            return

        # 1. Map the 'break/continue' stmt with an unique boolean variable V.
        variable_name = unique_name.generate(BREAK_NAME_PREFIX)

        # 2. Find the first ancestor block containing this 'break/continue', a
        # block can be a node containing stmt list. We should remove all stmts
        # after the 'break/continue' and set the V to True here.
        first_block_index = self._remove_stmts_after_break_continue(
            node, variable_name, loop_node_index)

        # 3. Add 'if V' for stmts in ancestor blocks between the first one
        # (exclusive) and the ancestor loop (inclusive)
        self._replace_if_stmt(loop_node_index, first_block_index, variable_name)

        # 4. For 'break' add break into condition of the loop.
        assign_false_node = gast.Assign(
            targets=[
                gast.Name(
                    id=variable_name,
                    ctx=gast.Store(),
                    annotation=None,
                    type_comment=None)
            ],
            value=gast.Constant(
                value=False, kind=None))
        self._add_stmt_before_cur_node(loop_node_index, assign_false_node)

        loop_node.test = gast.BoolOp(
            op=gast.And(),
            values=[
                loop_node.test, gast.Name(
                    id=variable_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None)
            ])

    def visit_Continue(self, node):
        loop_node_index = self._find_ancestor_loop_index(node)
        assert loop_node_index != -1, "SyntaxError: 'break' outside loop"
        loop_node = self.ancestor_nodes[loop_node_index]

        if not isinstance(loop_node, gast.While):
            # "For in range loop" will be transformed into while at
            # ForToWhileTransformer, other "For loop" is unsupported now
            # so we just do nothing and return
            return

        # 1. Map the 'break/continue' stmt with an unique boolean variable V.
        variable_name = unique_name.generate(BREAK_NAME_PREFIX)

        # 2. Find the first ancestor block containing this 'break/continue', a
        # block can be a node containing stmt list. We should remove all stmts
        # after the 'break/continue' and set the V to True here.
        first_block_index = self._remove_stmts_after_break_continue(
            node, variable_name, loop_node_index)

        # 3. Add 'if V' for stmts in ancestor blocks between the first one
        # (exclusive) and the ancestor loop (inclusive)
        self._replace_if_stmt(loop_node_index, first_block_index, variable_name)

        # 4. For 'continue', set continue to False at the beginning of each loop
        assign_false_node = gast.Assign(
            targets=[
                gast.Name(
                    id=variable_name,
                    ctx=gast.Store(),
                    annotation=None,
                    type_comment=None)
            ],
            value=gast.Constant(
                value=False, kind=None))
        loop_node.body.insert(0, assign_false_node)

    def _remove_stmts_after_break_continue(
            self, break_continue_node, break_continue_name, loop_node_index):
        for first_block_index in range(
                len(self.ancestor_nodes) - 1, loop_node_index - 1, -1):
            first_block = self.ancestor_nodes[first_block_index]
            if hasattr(first_block,
                       "body") and self._replace_break_conntinue_in_stmt_lists(
                           first_block.body, break_continue_node):
                return first_block_index

            if hasattr(first_block,
                       "orelse") and self._replace_break_conntinue_in_stmt_list(
                           first_block.orelse, break_continue_node):
                return first_block_index

        return first_block_index

    def _replace_break_conntinue_in_stmt_list(self, stmt_list,
                                              break_continue_node):
        i = index_in_list(stmt_list, break_continue_node)
        if i == -1:
            return False
        stmt_list = stmt_list[0:i]
        assign_true_node = gast.Assign(
            targets=[
                gast.Name(
                    id=break_continue_name,
                    ctx=gast.Store(),
                    annotation=None,
                    type_comment=None)
            ],
            value=gast.Constant(
                value=True, kind=None))
        stmt_list.append(assign_true_node)
        return True

    def _replace_if_stmt(self, loop_node_index, first_block_index,
                         break_continue_name):
        for i in range(first_block_index - 1, loop_node_index - 1, -1):
            cur_node = self.ancestor_nodes[i]
            son_node = self.ancestor_nodes[i + 1]
            if hasattr(cur_node,
                       'body') and self._replace_after_node_to_if_in_stmt_list(
                           cur_node.body, son_node, break_continue_name):
                continue
            if hasattr(
                    cur_node,
                    'orelse') and self._replace_after_node_to_if_in_stmt_list(
                        cur_node.orelse, son_node, break_continue_name):
                continue

    def _replace_after_node_to_if_in_stmt_list(self, stmt_list, node,
                                               break_continue_name):
        i = index_in_list(stmt_list, node)
        if i == -1:
            return False

        if i == len(stmt_list) - 1:
            # No need to add, we consider this as added successfully
            return True

        if_stmt = gast.If(test=gast.Name(
            id=break_continue_name,
            ctx=gast.Store(),
            annotation=None,
            type_comment=None),
                          body=[gast.Pass()],
                          orelse=stmt_list[i + 1:])
        stmt_list = stmt_list[0:i + 1]
        stmt_list.append(if_stmt)
        return True

    def _add_stmt_before_cur_node(self, cur_node_index, stmt_node):
        cur_node = self.ancestor_nodes[cur_node_index]
        parent_node = self.ancestor_nodes[cur_node_index - 1]
        if hasattr(parent_node,
                   "body") and self._add_stmt_into_list_before_node(
                       parent_node.body, cur_node, stmt_node):
            return True
        if hasattr(parent_node,
                   "orelse") and slef._add_stmt_into_list_before_node(
                       parent_node.orelse, cur_node, stmt_node):
            return True
        return False

    def _add_stmt_into_list_before_node(self, stmt_list, node, stmt_node):
        i = index_in_list(stmt_list, node)
        if i == -1:
            return False
        stmt_list.insert(i + 1, stmt_node)
        return True

    def _find_ancestor_loop_index(self, node):
        for i in range(len(self.ancestor_nodes) - 1, -1, -1):
            if isinstance(self.ancestor_nodes[i], (gast.For, gast.While)):
                return i
        return -1
