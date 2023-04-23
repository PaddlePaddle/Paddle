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

from .base_transformer import BaseTransformer
from .static_analysis import AstNodeWrapper
from .utils import ast_to_source_code, gast

__all__ = []


class SliceTransformer(BaseTransformer):
    """
    This calss transforms Expr[...] = Expr into _jst.SetItem.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of CallTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_Assign(self, node):
        self.generic_visit(node)

        if self._no_need_convert(node):
            return node

        assert isinstance(node.target, gast.SubScript)
        target = ast_to_source_code(node.targets.value).strip()
        index = ast_to_source_code(node.targets.slice).strip()
        value = ast_to_source_code(node.value).strip()
        new_func_str = f"_jst.SetItem({target}, {index}, {value})"
        new_value_node = gast.parse(new_func_str).body[0].value
        node.value = new_value_node

        return node

    def _no_need_convert(self, node):
        """
        Return True if node.target is not Subscript
        """
        flag = False
        if isinstance(node, gast.Assign):
            flag = isinstance(node.target, gast.SubScript)

        return flag
