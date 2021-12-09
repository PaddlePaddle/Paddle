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

import ast
from paddle.utils import gast
import sys
import textwrap
import unittest


class GastNodeTransformer(gast.NodeTransformer):
    def __init__(self, root):
        self.root = root

    def apply(self):
        return self.generic_visit(self.root)

    def visit_Name(self, node):
        """
        Param in func is ast.Name in PY2, but ast.arg in PY3.
        It will be generally represented by gast.Name in gast.
        """
        if isinstance(node.ctx, gast.Param) and node.id != "self":
            node.id += '_new'

        return node

    def visit_With(self, node):
        """
        The fileds `context_expr/optional_vars` of `ast.With` in PY2
        is moved into `ast.With.items.withitem` in PY3.
        It will be generally represented by gast.With.items.withitem in gast.
        """
        assert hasattr(node, 'items')
        if node.items:
            withitem = node.items[0]
            assert isinstance(withitem, gast.withitem)
            if isinstance(withitem.context_expr, gast.Call):
                func = withitem.context_expr.func
                if isinstance(func, gast.Name):
                    func.id += '_new'
        return node

    def visit_Call(self, node):
        """
        The fileds `starargs/kwargs` of `ast.Call` in PY2
        is moved into `Starred/keyword` in PY3.
        It will be generally represented by gast.Starred/keyword in gast.
        """
        assert hasattr(node, 'args')
        if node.args:
            assert isinstance(node.args[0], gast.Starred)
            # modify args
            if isinstance(node.args[0].value, gast.Name):
                node.args[0].value.id += '_new'

        assert hasattr(node, 'keywords')
        if node.keywords:
            assert isinstance(node.keywords[0], gast.keyword)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        """
        In PY3.8, ast.Num/Str/Bytes/None/False/True are merged into ast.Constant.
        But these types are still available and will be deprecated in future versions.
        ast.Num corresponds to gast.Num in PY2, and corresponds to gast.Constant in PY3.
        """
        if isinstance(node.value, int):
            node.value *= 2
        return node

    def visit_Num(self, node):
        """
        ast.Num is available before PY3.8, and see visit_Constant for details.
        """
        node.n *= 2
        return node

    def visit_Subscript(self, node):
        """
        Before PY3.8, the fields of ast.subscript keeps exactly same between PY2 and PY3.
        After PY3.8, the field `slice` with ast.Slice will be changed into ast.Index(Tuple).
        It will be generally represented by gast.Index or gast.Slice in gast.
        Note: Paddle doesn't support PY3.8 currently.
        """
        self.generic_visit(node)
        return node


def code_gast_ast(source):
    """
    Transform source_code into gast.Node and modify it,
    then back to ast.Node.
    """
    source = textwrap.dedent(source)
    root = gast.parse(source)
    new_root = GastNodeTransformer(root).apply()
    ast_root = gast.gast_to_ast(new_root)
    return ast.dump(ast_root)


def code_ast(source):
    """
    Transform source_code into ast.Node, then dump it.
    """
    source = textwrap.dedent(source)
    root = ast.parse(source)
    return ast.dump(root)


class TestPythonCompatibility(unittest.TestCase):
    def _check_compatibility(self, source, target):
        source_dump = code_gast_ast(source)
        target_dump = code_ast(target)
        self.assertEqual(source_dump, target_dump)

    def test_param_of_func(self):
        """
        Param in func is ast.Name in PY2, but ast.arg in PY3.
        It will be generally represented by ast.Name in gast.
        """
        source = """
            def foo(x, y):
                return x + y
        """
        target = """
            def foo(x_new, y_new):
                return x + y
        """
        self._check_compatibility(source, target)

    # The 0.3.3 version of gast has a bug in python3.8 that
    # would cause the following tests to fail. But this 
    # problem doesn't affect the use of Paddle's related 
    # functions, therefore, the following tests would be 
    # disable in python3.8.
    #
    # This problem had been fixed and updated to version 
    # 0.4.1 of gast.
    #
    # More information please refer to:
    # https://github.com/serge-sans-paille/gast/issues/49
    if sys.version_info < (3, 8):

        def test_with(self):
            """
            The fileds `context_expr/optional_vars` of `ast.With` in PY2
            is moved into `ast.With.items.withitem` in PY3.
            """
            source = """
            with guard():
                a = 1
            """
            target = """
            with guard_new():
                a = 1
            """
            self._check_compatibility(source, target)

        def test_subscript_Index(self):
            source = """
                x = y()[10]
            """
            target = """
                x = y()[20]
            """
            self._check_compatibility(source, target)

        def test_subscript_Slice(self):
            source = """
                x = y()[10:20]
            """
            target = """
                x = y()[20:40]
            """
            self._check_compatibility(source, target)

        def test_call(self):
            source = """
                y = foo(*arg)
            """
            target = """
                y = foo(*arg_new)
            """
            self._check_compatibility(source, target)


if __name__ == '__main__':
    unittest.main()
