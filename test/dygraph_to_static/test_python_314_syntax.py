# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import importlib.util
import pathlib
import sys
import tempfile
import textwrap
import unittest

import numpy as np

import paddle
from paddle.jit.dy2static.utils import ast_to_func
from paddle.utils import gast


def _template_string_source():
    return textwrap.dedent(
        '''
        def template_string_target(x):
            message = t"x = {x}"
            return x + 1
        '''
    )


def template_string_target(x):
    return x + 1


@unittest.skipIf(
    sys.version_info < (3, 14),
    "Python 3.14 t-string AST nodes are not available.",
)
class TestPython314Syntax(unittest.TestCase):
    def _template_node_from_source(self):
        ast_root = gast.parse(_template_string_source())
        template_node = ast_root.body[0].body[0].value
        return ast_root, template_node

    def test_gast_parse_keeps_template_string_nodes(self):
        _, template_node = self._template_node_from_source()

        self.assertIsInstance(template_node, gast.TemplateStr)
        self.assertEqual(
            [type(value) for value in template_node.values],
            [gast.Constant, gast.Interpolation],
        )
        self.assertEqual(template_node.values[0].value, "x = ")
        self.assertEqual(template_node.values[1].str, "x")

    def test_ast_to_func_keeps_template_string_executable(self):
        ast_root, template_node = self._template_node_from_source()
        self.assertIsInstance(template_node, gast.TemplateStr)

        transformed_func, _ = ast_to_func(
            ast_root, template_string_target, delete_on_exit=False
        )

        self.assertEqual(transformed_func(2), 3)

    def test_to_static_with_template_string_executes(self):
        source = _template_string_source()
        with tempfile.TemporaryDirectory() as tmp_dir:
            module_path = pathlib.Path(tmp_dir) / "template_string_module.py"
            module_path.write_text("import paddle\n" + source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(
                "template_string_module", module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            x = paddle.to_tensor([1.0, 2.0])
            actual = paddle.jit.to_static(module.template_string_target)(x)

        np.testing.assert_allclose(actual.numpy(), np.array([2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
