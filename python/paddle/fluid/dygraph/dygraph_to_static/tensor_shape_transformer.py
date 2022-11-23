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

from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.base_transformer import BaseTransformer


class TensorShapeTransformer(BaseTransformer):
    """
    This class transforms variable.shape  into Static Graph Ast.
    All 'xxx.shape' will be converted int '_jst.Shape(x)'.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of TensorShapeTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == 'shape':
            args = ast_to_source_code(node.value).strip()
            # NOTE(dev): we can deal with paddle.shape in this case, but it's
            # not pretty to modify into 'convert_shape(paddle)(x)[0]'.
            if args != 'paddle':
                convert_shape_func = "_jst.Shape({})".format(args)
                shape_node = gast.parse(convert_shape_func).body[0].value
                return shape_node
        return node
