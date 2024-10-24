#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TypeHintTransformer(BaseTransformer):
    """
    A class remove all the typehint in gast.Name(annotation).
    Please put it behind other transformers because other transformer may relay on typehints.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        node.annotation = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        assign_node = gast.Assign(
            targets=[node.target],
            value=node.value,
            type_comment=None,
        )
        return assign_node
