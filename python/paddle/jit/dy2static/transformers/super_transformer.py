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

from paddle.utils import gast

from .base import BaseTransformer

__all__ = []


class SuperTransformer(BaseTransformer):
    """
    This class transforms super() into super(__class__, <first argument>).
    """

    def __init__(self, root):
        self.root = root
        self.first_arg = None

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        if self.first_arg is not None:
            return self.generic_visit(node)

        positional_args = node.args.posonlyargs + node.args.args
        if not positional_args:
            return self.generic_visit(node)

        self.first_arg = positional_args[0].id

        return self.generic_visit(node)

    def visit_Call(self, node):
        # super() -> _jst.WrapSuper(super)(x.__class__, x)
        self.generic_visit(node)
        if self.first_arg is None:
            return node
        if not isinstance(node.func, gast.Name):
            return node
        if node.func.id != "super":
            return node
        if node.args:
            return node

        new_fn_call_str = f"_jst.WrapSuper(super)({self.first_arg}.__class__, {self.first_arg})"
        new_fn_call_ast = gast.parse(new_fn_call_str).body[0]
        return new_fn_call_ast
