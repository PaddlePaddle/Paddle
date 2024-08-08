# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ast
import sys

import astor

from paddle.utils import gast


def ast_to_source_code(ast_node):
    """
    Transforms ast node into source code.
    """
    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            f"Type of ast_root should be gast.AST or ast.AST, but received {type(ast_node)}."
        )
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)

    if sys.version_info >= (3, 9):
        ast.fix_missing_locations(ast_node)
        return ast.unparse(ast_node)

    # Do not wrap lines even if they are too long
    def pretty_source(source):
        return ''.join(source)

    source_code = astor.to_source(ast_node, pretty_source=pretty_source)
    return source_code
