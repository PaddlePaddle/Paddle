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
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.utils import FunctionNameLivenessAnalysis
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import create_undefined_var
from paddle.fluid.dygraph.dygraph_to_static.base_transformer import BaseTransformer


class CreateVariableTransformer(BaseTransformer):
    """
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Type of input node should be AstNodeWrapper, but received %s ." % type(
            wrapper_root)
        self.root = wrapper_root.node
        FunctionNameLivenessAnalysis(self.root)

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        #attributes = set(filter(lambda x: '.' in x, node.pd_scope.modified_vars()))
        self.generic_visit(node)
        bodys = node.body
        names = sorted(node.pd_scope.created_vars())
        for name in names:
            bodys[0:0] = [create_undefined_var(name)]
        return node
