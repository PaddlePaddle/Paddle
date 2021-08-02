# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import ast
import gast


def _generate_translators(to):
    class Translator(ast.NodeTransformer):
        def _visit(self, node):
            if isinstance(node, list):
                return [self._visit(n) for n in node]
            elif isinstance(node, ast.AST):
                return self.visit(node)
            else:
                return node

        def generic_visit(self, node):
            cls = type(node).__name__
            # handle nodes that are not part of the AST
            if not hasattr(to, cls):
                return
            new_node = getattr(to, cls)()
            for field in node._fields:
                setattr(new_node, field, self._visit(getattr(node, field)))
            for attr in getattr(node, '_attributes'):
                if hasattr(node, attr):
                    setattr(new_node, attr, getattr(node, attr))
            return new_node

    return Translator


AstToGAst = _generate_translators(gast)

GAstToAst = _generate_translators(ast)
