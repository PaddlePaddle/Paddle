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
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import NodeVarType
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor


class AssertTransformer(gast.NodeTransformer):
    """
    A class transforms python assert to fluid.layers.Assert.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of AssertTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)

    def transform(self):
        self.visit(self.root)

    def visit_Assert(self, node):
        if not self.static_analysis_visitor.is_tensor_node(node.test):
            return node
        cast_node = gast.Call(
            func=gast.parse("fluid.layers.cast").body[0].value,
            args=[node.test, gast.Constant(
                value="bool", kind=None)],
            keywords=[])
        assert_node = gast.Call(
            func=gast.parse("fluid.layers.Assert").body[0].value,
            args=[cast_node],
            keywords=[])
        return gast.Expr(value=assert_node)
