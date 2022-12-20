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

from paddle.jit.dy2static.static_analysis import AstNodeWrapper
from paddle.jit.dy2static.utils import ast_to_source_code, is_paddle_api
from paddle.utils import gast

from .base_transformer import BaseTransformer

PDB_SET = "pdb.set_trace"

__all__ = []


class CallTransformer(BaseTransformer):
    """
    This class transforms function calls into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of CallTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def _no_need_convert_call(self, node):
        """
        Determines whether a function needs to be transformed by `convert_call`.
        It doesn't need to be transformed when a function satisfies the following conditions:
          1. It's a api of paddle
          2. It's a python builtin function not include `len`, `zip`, `range` and `enumerate`
        """
        assert isinstance(node, gast.Call)
        if is_paddle_api(node):
            return True

        func_str = ast_to_source_code(node.func).strip()
        try:
            from paddle.jit.dy2static.convert_call_func import is_builtin

            need_convert_builtin_func_list = {
                'len',
                'zip',
                'range',
                'enumerate',
                'print',
            }
            is_builtin = eval("is_builtin({})".format(func_str))  # noqa: F811
            need_convert = func_str in need_convert_builtin_func_list
            return is_builtin and not need_convert
        except Exception:
            return False

    def transform(self):
        self.visit(self.root)

    def visit_Call(self, node):
        self.generic_visit(node)

        if self._no_need_convert_call(node):
            return node

        func_str = ast_to_source_code(node.func).strip()

        # NOTE(liym27): Don't convert `pad.set_trace` even if the convertion doesn't work finally, because
        # it is clearer to see where it is called from.
        if PDB_SET in func_str:
            return node

        new_func_str = "_jst.Call({})".format(func_str)
        new_func_ast = gast.parse(new_func_str).body[0].value
        node.func = new_func_ast

        return node
