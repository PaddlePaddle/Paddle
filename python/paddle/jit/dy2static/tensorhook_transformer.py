# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import collections

from paddle.utils import gast

from .base_transformer import BaseTransformer


class RegisterHookTransformer(BaseTransformer):
    def __init__(self, root):
        self.register_hook_pos_map = collections.defaultdict(list)
        self.assignment_pos_map = collections.defaultdict(list)
        self.root = root

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def visit_FunctionDef(self, func_def):
        # The inner function that has register_hook will not be processed
        check_register_hook = next(
            (
                node
                for node in gast.walk(func_def)
                if isinstance(node, gast.Attribute)
                and node.attr == 'register_hook'
            ),
            None,
        )
        if check_register_hook is None:
            return func_def

        register_hook_pos_map = self.register_hook_pos_map
        assignment_pos_map = self.assignment_pos_map

        for i in range(len(func_def.body) - 1, -1, -1):

            body = func_def.body[i]
            # Check if the code body contains the register_hook
            if isinstance(body, gast.Expr):
                for node in gast.walk(body):
                    if (
                        isinstance(node, gast.Attribute)
                        and node.attr == 'register_hook'
                    ):
                        # parameter name for register_hook
                        param_name = node.value.id
                        register_hook_pos_map[param_name].append(i)
            elif isinstance(body, gast.Assign):
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
        return func_def
