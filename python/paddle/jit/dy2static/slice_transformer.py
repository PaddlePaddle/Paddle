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

import paddle
from paddle.fluid.framework import Variable

from .base_transformer import BaseTransformer
from .static_analysis import AstNodeWrapper
from .utils import ast_to_source_code, gast

__all__ = []


class GetSetter:
    """
    GetSetter is a Proxy class implenenting Python getter / setter magic method. It's basic unit
    for _jst.GSet(...) to make us not care what ast is like.
    """

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, item):
        return self.obj[item]

    def __setitem__(self, item, value):
        if isinstance(self.obj, Variable):
            new_var = paddle.fluid.framework._setitem_impl_(
                self.obj, item, value
            )
            # NOTE(dev): Update __dict__ will not modify the id(), but only move the
            # pointed reference object to the new one.
            self.obj.__dict__.update(new_var.__dict__)
        elif hasattr(self.obj, '__setitem__'):
            self.obj.__setitem__(item, value)
        else:
            raise RuntimeError(
                "Unsupport _jst.GSet for {} because it has no __setitem__ method.".format(
                    self.obj
                )
            )


class SliceTransformer(BaseTransformer):
    """
    This calss transforms Expr[...] = Expr into _jst.GSet(Expr)[...] = Expr.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of CallTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_Subscript(self, node):
        self.generic_visit(node)

        value = ast_to_source_code(node.value).strip()
        new_value_str = f"_jst.GSet({value})"
        new_value_node = gast.parse(new_value_str).body[0].value
        node.value = new_value_node

        return node
