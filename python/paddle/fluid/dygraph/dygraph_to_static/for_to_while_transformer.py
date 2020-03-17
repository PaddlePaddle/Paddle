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

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.utils import get_constant_variable_node


class ForToWhileTransformer(gast.NodeTransformer):
    """
    Transform python for loop into while loop, to make it easier for
    fluid.layers.while_loop
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input is NOT AstNodeWrapper for the initialization of ForToWhileTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit(self, node):
        self.generic_visit(node)
        # All parent nodes that may contain gast.For
        if hasattr(node, 'body'):
            self.replace_stmt_list(node.body)
        if hasattr(node, 'orelse'):
            self.replace_stmt_list(node.orelse)
        return node

    def replace_stmt_list(self, body_list):
        if not isinstance(body_list, list):
            return

        i = 0
        while i < len(body_list):
            if isinstance(body_list[i], gast.For):
                new_stmts = self.get_for_stmt_nodes(body_list[i])
                body_list[i:i + 1] = new_stmts
                i += len(new_stmts)
            else:
                i += 1

    def get_for_range_node(self, node):
        if not isinstance(node.iter, gast.Call):
            return None
        if not isinstance(node.iter.func, gast.Name):
            return None
        if node.iter.func.id != "range":
            return None
        return node.iter

    def get_for_args_stmts(self, iter_name, args_list):
        '''
        Returns 3 gast stmt nodes for argument.
        1. Initailize of iterate variable
        2. Condition for the loop
        3. Statement for changing of iterate variable during the loop
        NOTE(TODO): Python allows to access iteration variable after loop, such
           as "for i in range(10)" will create i = 9 after the loop. But using
           current conversion will make i = 10. We should find a way to change it
        '''
        len_range_args = len(args_list)
        assert len_range_args >= 1 and len_range_args <= 3, "range() function takes 1 to 3 arguments"
        if len_range_args == 1:
            init_stmt = get_constant_variable_node(iter_name, 0)
        else:
            init_stmt = gast.Assign(
                targets=[
                    gast.Name(
                        id=iter_name,
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None)
                ],
                value=args_list[0])

        range_max_node = args_list[0] if len_range_args == 1 else args_list[1]
        step_node = args_list[2] if len_range_args == 3 else gast.Constant(
            value=1, kind=None)

        cond_stmt = gast.Compare(
            left=gast.BinOp(
                left=gast.Name(
                    id=iter_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None),
                op=gast.Add(),
                right=step_node),
            ops=[gast.LtE()],
            comparators=[range_max_node])

        change_stmt = gast.AugAssign(
            target=gast.Name(
                id=iter_name,
                ctx=gast.Store(),
                annotation=None,
                type_comment=None),
            op=gast.Add(),
            value=step_node)

        return init_stmt, cond_stmt, change_stmt

    def get_for_stmt_nodes(self, node):
        assert isinstance(
            node, gast.For), "Input node is NOT gast.For in get_for_stmt_nodes"

        # TODO: support non-range case
        range_call_node = self.get_for_range_node(node)
        if range_call_node is None:
            return [node]

        if not isinstance(node.target, gast.Name):
            return [node]
        iter_var_name = node.target.id

        init_stmt, cond_stmt, change_stmt = self.get_for_args_stmts(
            iter_var_name, range_call_node.args)

        new_body = node.body
        new_body.append(change_stmt)
        while_node = gast.While(
            test=cond_stmt, body=new_body, orelse=node.orelse)
        return [init_stmt, while_node]
