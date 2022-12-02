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

import unittest

from paddle.jit.dy2static.utils import ast_to_source_code
from paddle.jit.dy2static.variable_trans_func import create_fill_constant_node


class TestVariableTransFunc(unittest.TestCase):
    def test_create_fill_constant_node(self):
        node = create_fill_constant_node("a", 1.0)
        source = "a = paddle.full(shape=[1], dtype='float64', fill_value=1.0, name='a')"
        self.assertEqual(
            ast_to_source_code(node).replace('\n', '').replace(' ', ''),
            source.replace(' ', ''),
        )

        node = create_fill_constant_node("b", True)
        source = "b = paddle.full(shape=[1], dtype='bool', fill_value=True, name='b')"
        self.assertEqual(
            ast_to_source_code(node).replace('\n', '').replace(' ', ''),
            source.replace(' ', ''),
        )

        node = create_fill_constant_node("c", 4293)
        source = "c = paddle.full(shape=[1], dtype='int64', fill_value=4293, name='c')"
        self.assertEqual(
            ast_to_source_code(node).replace('\n', '').replace(' ', ''),
            source.replace(' ', ''),
        )

        self.assertIsNone(create_fill_constant_node("e", None))
        self.assertIsNone(create_fill_constant_node("e", []))


if __name__ == '__main__':
    unittest.main()
