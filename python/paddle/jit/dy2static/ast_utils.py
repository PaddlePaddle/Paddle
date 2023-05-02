# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ast
import collections
import inspect
import textwrap

import astor

from paddle.utils import gast


def ast_to_source_code(ast_node):
    """
    Transforms ast node into source code.
    """
    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s."
            % type(ast_node)
        )
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)

    # Do not wrap lines even if they are too long
    def pretty_source(source):
        return ''.join(source)

    source_code = astor.to_source(ast_node, pretty_source=pretty_source)
    return source_code


class RegisterHookVisitor(gast.NodeVisitor):
    def __init__(self, func_name):
        self.register_hook_pos_map = collections.defaultdict(list)
        self.assignment_pos_map = collections.defaultdict(list)
        self.func_name = func_name

    def visit_FunctionDef(self, func_def):
        # The inner function that has register_hook will not be processed
        if func_def.name != self.func_name:
            return
        register_hook_pos_map = self.register_hook_pos_map
        assignment_pos_map = self.assignment_pos_map

        for i in range(len(func_def.body) - 1, -1, -1):

            body = func_def.body[i]
            # Check if the code body contains the register_hook
            if isinstance(body, ast.Expr):
                for node in ast.walk(body):
                    if (
                        isinstance(node, ast.Attribute)
                        and node.attr == 'register_hook'
                    ):
                        # parameter name for register_hook
                        param_name = node.value.id
                        register_hook_pos_map[param_name].append(i)
            elif isinstance(body, ast.Assign):
                for target in body.targets:
                    assignment_pos_map[target.id].append(i)

        # Confirm the order
        order_map = {}
        for k, idx_list in register_hook_pos_map.items():
            for idx in idx_list:
                if k not in assignment_pos_map:
                    order_map[idx] = 1
                else:
                    for assignment_idx in assignment_pos_map[k]:
                        if idx > assignment_idx:
                            order_map[idx] = assignment_idx + 1
                            break
        code_order = [*range(len(func_def.body))]
        for k, v in sorted(order_map.items(), key=lambda x: x[1], reverse=True):
            if k == v:
                continue
            code_order.remove(k)
            code_order.insert(v, k)

        # rearrange the code according to the specified order
        new_body = [func_def.body[i] for i in code_order]
        func_def.body = new_body


def modify_function_code(func):
    """
    Modify the function code for the register hook
    """

    func_ast = ast.parse(textwrap.dedent(inspect.getsource(func)))
    # check if there is register_hook on code after visit the tree.
    check_register_hook = next(
        (
            node
            for node in ast.walk(func_ast)
            if isinstance(node, ast.Attribute) and node.attr == 'register_hook'
        ),
        None,
    )
    if check_register_hook is None:
        return

    visitor = RegisterHookVisitor(func.__name__)
    visitor.visit(func_ast)

    def pretty_source(source):
        return ''.join(source)

    new_code = astor.to_source(func_ast, pretty_source=pretty_source)
    return new_code
