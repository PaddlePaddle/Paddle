# Copyright (c) 2016, Serge Guelton
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 	Redistributions of source code must retain the above copyright notice, this
# 	list of conditions and the following disclaimer.

# 	Redistributions in binary form must reproduce the above copyright notice,
# 	this list of conditions and the following disclaimer in the documentation
# 	and/or other materials provided with the distribution.

# 	Neither the name of HPCProject, Serge Guelton nor the names of its
# 	contributors may be used to endorse or promote products derived from this
# 	software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# NOTE(paddle-dev): We introduce third-party library Gast as unified AST
# representation. See https://github.com/serge-sans-paille/gast for details.

from .astn import AstToGAst, GAstToAst
from . import gast
import ast
import sys


class Ast3ToGAst(AstToGAst):
    if sys.version_info.minor < 10:

        def visit_alias(self, node):
            new_node = gast.alias(
                self._visit(node.name),
                self._visit(node.asname),
            )
            new_node.lineno = new_node.col_offset = None
            new_node.end_lineno = new_node.end_col_offset = None
            return new_node

    def visit_Name(self, node):
        new_node = gast.Name(
            node.id,  # micro-optimization here, don't call self._visit
            self._visit(node.ctx),
            None,
            None,
        )
        return ast.copy_location(new_node, node)

    def visit_arg(self, node):
        extra_arg = self._visit(node.type_comment)

        new_node = gast.Name(
            node.arg,  # micro-optimization here, don't call self._visit
            gast.Param(),
            self._visit(node.annotation),
            extra_arg,  # type_comment
        )
        return ast.copy_location(new_node, node)

    def visit_ExceptHandler(self, node):
        if node.name:
            new_node = gast.ExceptHandler(
                self._visit(node.type),
                gast.Name(node.name, gast.Store(), None, None),
                self._visit(node.body),
            )
            return ast.copy_location(new_node, node)
        else:
            return self.generic_visit(node)

    if 8 <= sys.version_info.minor < 12:

        def visit_ClassDef(self, node):
            new_node = gast.ClassDef(
                self._visit(node.name),
                self._visit(node.bases),
                self._visit(node.keywords),
                self._visit(node.body),
                self._visit(node.decorator_list),
                [],  # type_params
            )
            return gast.copy_location(new_node, node)

        def visit_FunctionDef(self, node):
            new_node = gast.FunctionDef(
                self._visit(node.name),
                self._visit(node.args),
                self._visit(node.body),
                self._visit(node.decorator_list),
                self._visit(node.returns),
                self._visit(node.type_comment),
                [],  # type_params
            )
            return gast.copy_location(new_node, node)

        def visit_AsyncFunctionDef(self, node):
            new_node = gast.AsyncFunctionDef(
                self._visit(node.name),
                self._visit(node.args),
                self._visit(node.body),
                self._visit(node.decorator_list),
                self._visit(node.returns),
                self._visit(node.type_comment),
                [],  # type_params
            )
            return gast.copy_location(new_node, node)


class GAstToAst3(GAstToAst):
    if sys.version_info.minor < 10:

        def visit_alias(self, node):
            new_node = ast.alias(
                self._visit(node.name), self._visit(node.asname)
            )
            return new_node

    def visit_Assign(self, node):
        new_node = ast.Assign(
            self._visit(node.targets),
            self._visit(node.value),
        )

        return ast.copy_location(new_node, node)

    def _make_arg(self, node):
        if node is None:
            return None

        extra_args = (self._visit(node.type_comment),)

        new_node = ast.arg(
            self._visit(node.id), self._visit(node.annotation), *extra_args
        )
        return ast.copy_location(new_node, node)

    def visit_Name(self, node):
        new_node = ast.Name(
            self._visit(node.id),
            self._visit(node.ctx),
        )
        return ast.copy_location(new_node, node)

    def visit_ExceptHandler(self, node):
        if node.name:
            new_node = ast.ExceptHandler(
                self._visit(node.type), node.name.id, self._visit(node.body)
            )
            return ast.copy_location(new_node, node)
        else:
            return self.generic_visit(node)

    if 5 <= sys.version_info.minor < 12:

        def visit_ClassDef(self, node):
            new_node = ast.ClassDef(
                self._visit(node.name),
                self._visit(node.bases),
                self._visit(node.keywords),
                self._visit(node.body),
                self._visit(node.decorator_list),
            )
            return ast.copy_location(new_node, node)

    if 8 <= sys.version_info.minor < 12:

        def visit_FunctionDef(self, node):
            new_node = ast.FunctionDef(
                self._visit(node.name),
                self._visit(node.args),
                self._visit(node.body),
                self._visit(node.decorator_list),
                self._visit(node.returns),
                self._visit(node.type_comment),
            )
            return ast.copy_location(new_node, node)

        def visit_AsyncFunctionDef(self, node):
            new_node = ast.AsyncFunctionDef(
                self._visit(node.name),
                self._visit(node.args),
                self._visit(node.body),
                self._visit(node.decorator_list),
                self._visit(node.returns),
                self._visit(node.type_comment),
            )
            return ast.copy_location(new_node, node)

    def visit_arguments(self, node):
        extra_args = [
            self._make_arg(node.vararg),
            [self._make_arg(n) for n in node.kwonlyargs],
            self._visit(node.kw_defaults),
            self._make_arg(node.kwarg),
            self._visit(node.defaults),
        ]
        if sys.version_info.minor >= 8:
            new_node = ast.arguments(
                [self._make_arg(arg) for arg in node.posonlyargs],
                [self._make_arg(n) for n in node.args],
                *extra_args,
            )
        else:
            new_node = ast.arguments(
                [self._make_arg(n) for n in node.args], *extra_args
            )
        return new_node


def ast_to_gast(node):
    return Ast3ToGAst().visit(node)


def gast_to_ast(node):
    return GAstToAst3().visit(node)
