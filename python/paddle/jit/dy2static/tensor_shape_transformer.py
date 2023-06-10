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

from .base_transformer import BaseTransformer
from .utils import ast_to_source_code

__all__ = []


class TensorShapeTransformer(BaseTransformer):
    """
    This class transforms variable.shape  into Static Graph Ast.
    All 'xxx.shape' will be converted int '_jst.Shape(x)'.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        self.visit(self.root)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == 'shape':
            args = ast_to_source_code(node.value).strip()
            # NOTE(dev): we can deal with paddle.shape in this case, but it's
            # not pretty to modify into 'convert_shape(paddle)(x)[0]'.
            if args != 'paddle':
                convert_shape_func = f"_jst.Shape({args})"
                shape_node = gast.parse(convert_shape_func).body[0].value
                return shape_node
        return node
