#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import ast

from .static_analysis import AstNodeWrapper, StaticAnalysisVisitor

__all__ = ['DygraphToStaticAst']


class DygraphToStaticAst(ast.NodeTransformer):
    """
    Main class to transform Dygraph to Static Graph
    """

    def get_static_ast(self, root):
        # save root for some analysis may need global AST 
        self.root = root
        self.static_analysis_root = StaticAnalysisVisitor(
            root).get_node_wrapper_root()
        self.transfer_from_node_type(self.static_analysis_root)
        return self.static_analysis_root

    def transfer_from_node_type(self, node):
        print("Not implemented")
