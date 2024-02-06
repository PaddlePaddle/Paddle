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


class BasicApiTransformer(BaseTransformer):
    """
    Class to transform basic API from dygraph to static graph.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        to_tensor_transformer = ToTensorTransformer(self.root)
        to_tensor_transformer.transform()
        attribute_transformer = AttributeJstTransformer(self.root)
        attribute_transformer.transform()
        self.visit(self.root)
        return self.root


class ToTensorTransformer(BaseTransformer):
    """
    Class to transform paddle.to_tensor and paddle.to_variable to paddle.assign
    """

    def __init__(self, node):
        assert isinstance(
            node, gast.AST
        ), "Input non-gast.AST node for the initialization of ToTensorTransformer."
        self.root = node

    def transform(self):
        self.visit(self.root)
        return self.root

    def visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if is_to_variable(node):
            node = to_assign_node(node)
        self.generic_visit(node)
        return node


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
        Can't convert name of function call, bacause this will affect CallTransformer.
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

    because `size` have different behavier when in dygraph / static graph mode
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


def is_to_variable(node):
    assert isinstance(node, gast.Call)
    api_name = ast_to_source_code(node.func).strip()

    return api_name.split(".")[-1] == "to_variable"


def to_assign_node(node):
    # Transform dygraph api `base.dygraph.to_variable` alias `paddle.to_tensor` to static api `paddle.assign`.
    # NOTE:
    #   1. Api `to_variable` supports data type {float16, float32, float64, int16, int32, int64, uint8, uint16},
    #   but api `assign` only supports {float32, float64, int32, int64, bool};
    #   2. If the input of api `assign` is numpy.ndarray, its size cannot be greater than 1024 * 1024.

    assert isinstance(node, gast.Call)
    assign_api = gast.parse('paddle.assign').body[0].value
    node.func = assign_api

    if node.args:
        node.args = [node.args[0]]
        node.keywords = []
    else:
        for idx, kw in enumerate(node.keywords):
            if kw.arg == 'value' or kw.arg == 'data':
                node.keywords[idx].arg = 'x'
                node.keywords = [node.keywords[idx]]
                node.args = []
                break
    return node
