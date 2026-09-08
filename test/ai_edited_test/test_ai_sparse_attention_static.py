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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.sparse_attention
# Target: cover uncovered lines 159-179 (static graph path)
# in paddle/nn/functional/sparse_attention.py

"""
测试模块：paddle.nn.functional.sparse_attention
Test Module: paddle.nn.functional.sparse_attention

本测试覆盖以下功能：
This test covers:
1. sparse_attention - 静态图路径（LayerHelper 创建变量和 op）
2. sparse_attention - 参数验证
3. sparse_attention - 导入验证
"""

import unittest

import paddle


class TestSparseAttentionExtendedImport(unittest.TestCase):
    """测试 sparse_attention 导入
    Test sparse_attention import"""

    def test_import_sparse_attention_func(self):
        """测试 sparse_attention 函数可导入
        Test sparse_attention function is importable"""
        from paddle.nn.functional.sparse_attention import sparse_attention

        self.assertTrue(callable(sparse_attention))


class TestSparseAttentionStaticGraphPath(unittest.TestCase):
    """测试 sparse_attention 静态图路径
    Test sparse_attention static graph path"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_sparse_attention_static_graph_build(self):
        """测试静态图模式下构建 sparse_attention op
        Test building sparse_attention op in static graph mode"""
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            query = paddle.static.data(
                name='query', shape=[1, 1, 4, 2], dtype='float32'
            )
            key = paddle.static.data(
                name='key', shape=[1, 1, 4, 2], dtype='float32'
            )
            value = paddle.static.data(
                name='value', shape=[1, 1, 4, 2], dtype='float32'
            )
            offset = paddle.static.data(
                name='offset', shape=[1, 1, 5], dtype='int32'
            )
            columns = paddle.static.data(
                name='columns', shape=[1, 1, 8], dtype='int32'
            )

            from paddle.nn.functional.sparse_attention import sparse_attention

            output = sparse_attention(query, key, value, offset, columns)

            self.assertIsNotNone(output)
            # 验证静态图模式下 op 被正确构建
            # Verify op is correctly built in static graph mode
            self.assertTrue(len(main_prog.global_block().ops) > 0)

    def test_sparse_attention_static_with_masks(self):
        """测试静态图模式下带 mask 的 sparse_attention
        Test sparse_attention with masks in static graph mode"""
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            query = paddle.static.data(
                name='query', shape=[1, 1, 4, 2], dtype='float32'
            )
            key = paddle.static.data(
                name='key', shape=[1, 1, 4, 2], dtype='float32'
            )
            value = paddle.static.data(
                name='value', shape=[1, 1, 4, 2], dtype='float32'
            )
            offset = paddle.static.data(
                name='offset', shape=[1, 1, 5], dtype='int32'
            )
            columns = paddle.static.data(
                name='columns', shape=[1, 1, 8], dtype='int32'
            )
            key_padding_mask = paddle.static.data(
                name='key_padding_mask', shape=[1, 4], dtype='float32'
            )
            attn_mask = paddle.static.data(
                name='attn_mask', shape=[4, 4], dtype='float32'
            )

            from paddle.nn.functional.sparse_attention import sparse_attention

            output = sparse_attention(
                query,
                key,
                value,
                offset,
                columns,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

            self.assertIsNotNone(output)

    def test_sparse_attention_static_output_vars(self):
        """测试静态图模式下输出变量
        Test output variables in static graph mode"""
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            query = paddle.static.data(
                name='query', shape=[1, 1, 4, 2], dtype='float32'
            )
            key = paddle.static.data(
                name='key', shape=[1, 1, 4, 2], dtype='float32'
            )
            value = paddle.static.data(
                name='value', shape=[1, 1, 4, 2], dtype='float32'
            )
            offset = paddle.static.data(
                name='offset', shape=[1, 1, 5], dtype='int32'
            )
            columns = paddle.static.data(
                name='columns', shape=[1, 1, 8], dtype='int32'
            )

            from paddle.nn.functional.sparse_attention import sparse_attention

            output = sparse_attention(query, key, value, offset, columns)

            # 验证输出变量被正确创建
            # Verify output variables are correctly created
            self.assertIsNotNone(output)
            # 验证图中有 ops
            # Verify graph has ops
            self.assertTrue(len(main_prog.global_block().ops) > 0)


class TestSparseAttentionDocstring(unittest.TestCase):
    """测试 sparse_attention 文档
    Test sparse_attention documentation"""

    def test_function_has_docstring(self):
        """测试函数有文档字符串
        Test function has docstring"""
        from paddle.nn.functional.sparse_attention import sparse_attention

        self.assertIsNotNone(sparse_attention.__doc__)
        self.assertIn("sparse", sparse_attention.__doc__.lower())
        self.assertIn("attention", sparse_attention.__doc__.lower())

    def test_docstring_contains_args(self):
        """测试文档包含参数说明
        Test docstring contains argument descriptions"""
        from paddle.nn.functional.sparse_attention import sparse_attention

        doc = sparse_attention.__doc__
        self.assertIn("query", doc.lower())
        self.assertIn("key", doc.lower())
        self.assertIn("value", doc.lower())


class TestSparseAttentionNameParam(unittest.TestCase):
    """测试 sparse_attention name 参数
    Test sparse_attention name parameter"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_name_parameter(self):
        """测试 name 参数传递
        Test name parameter passing"""
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            query = paddle.static.data(
                name='query', shape=[1, 1, 4, 2], dtype='float32'
            )
            key = paddle.static.data(
                name='key', shape=[1, 1, 4, 2], dtype='float32'
            )
            value = paddle.static.data(
                name='value', shape=[1, 1, 4, 2], dtype='float32'
            )
            offset = paddle.static.data(
                name='offset', shape=[1, 1, 5], dtype='int32'
            )
            columns = paddle.static.data(
                name='columns', shape=[1, 1, 8], dtype='int32'
            )

            from paddle.nn.functional.sparse_attention import sparse_attention

            output = sparse_attention(
                query, key, value, offset, columns, name="my_sparse_attn"
            )
            self.assertIsNotNone(output)


if __name__ == '__main__':
    unittest.main()
