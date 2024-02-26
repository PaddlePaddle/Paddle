# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ..utils import ast_to_source_code
from .base import BaseTransformer

__all__ = []


class NameloadJstTransformer(BaseTransformer):
    """
    change name and attribute load to __jst.Ld(name) pattern.
    for example:
        a.dtype -->  __jst.Ld(__jst.Ld(a).dtype)

    In paddle science and deepxde, we have to support changing tensor into variable
    in arbitrary occasion such as global tensor.

    NOTE: we only deal with ctx=Load() case.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        self.visit(self.root)
        return self.root

    def _surround_with_ld(self, node):
        node = (
            gast.parse(f"_jst.Ld({ast_to_source_code(node).strip()})")
            .body[0]
            .value
        )
        return node

    def visit_Call(self, node):
        """
        Can't convert name of function call, because this will affect CallTransformer.
        """
        node.args = [self.visit(arg) for arg in node.args]
        for keyword in node.keywords:
            keyword.value = self.visit(keyword.value)
        node.func = self.visit(node.func)
        return node

    def create_visit_with_convert_load(self, node_type, skip_fn=None):
        def visit(node):
            assert isinstance(node, node_type)
            if skip_fn and skip_fn(node):
                return node
            self.generic_visit(node)
            if isinstance(node.ctx, gast.Load):
                node = self._surround_with_ld(node)
            return node

        return visit

    def visit_Attribute(self, node):
        def skip_fn(node):
            if isinstance(node.value, gast.Name) and node.value.id == "_jst":
                return True
            return False

        return self.create_visit_with_convert_load(gast.Attribute, skip_fn)(
            node
        )

    def visit_Subscript(self, node):
        return self.create_visit_with_convert_load(gast.Subscript)(node)

    def visit_Name(self, node):
        return self.create_visit_with_convert_load(gast.Name)(node)


class AttributeJstTransformer(BaseTransformer):
    """
    change some special attribute into __jst.XXX(obj, "attr_name") format.
    for example:
        a.size  -->  __jst.attr(a, "size")

    because `size` have different behavior when in dygraph / static graph mode
    NOTE: we only deal with ctx=Load() case.
    """

    def __init__(self, node):
        assert isinstance(
            node, gast.AST
        ), "Input non-gast.AST node for the initialization of ToTensorTransformer."
        self.interested_name = {
            'size',
        }
        self.root = node

    def transform(self):
        self.visit(self.root)
        return self.root

    def visit_Attribute(self, node):
        assert isinstance(node, gast.Attribute)
        assert isinstance(node.attr, str)
        if (
            isinstance(node.ctx, gast.Load)
            and node.attr in self.interested_name
        ):
            attr = node.attr
            value = node.value
            node = (
                gast.parse(
                    f"_jst.Attr({ast_to_source_code(value).strip()}, \"{attr}\")"
                )
                .body[0]
                .value
            )
        self.generic_visit(node)
        return node
