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

# [AUTO-GENERATED] Test file for paddle.tensor.search operations
# 覆盖模块: paddle/tensor/search.py
# Uncovered lines: argmax, argmin, argsort, topk, where, index_select,
#   masked_select, nonzero

import unittest

import numpy as np

import paddle


class TestArgmax(unittest.TestCase):
    """测试 argmax 函数
    Test argmax function"""

    def test_argmax_default(self):
        """测试默认 argmax（展平后最大值索引）
        Test default argmax (flattened)"""
        x = paddle.to_tensor([[1, 3, 2], [4, 0, 5]])
        result = paddle.argmax(x)
        self.assertEqual(result.item(), 5)

    def test_argmax_axis(self):
        """测试指定轴 argmax
        Test argmax with axis"""
        x = paddle.to_tensor([[1, 3, 2], [4, 0, 5]])
        result = paddle.argmax(x, axis=1)
        np.testing.assert_array_equal(result.numpy(), np.array([1, 2]))

    def test_argmax_keepdim(self):
        """测试 keepdim 的 argmax
        Test argmax with keepdim"""
        x = paddle.randn([3, 4])
        result = paddle.argmax(x, axis=1, keepdim=True)
        self.assertEqual(result.shape, [3, 1])


class TestArgmin(unittest.TestCase):
    """测试 argmin 函数
    Test argmin function"""

    def test_argmin_default(self):
        """测试默认 argmin
        Test default argmin"""
        x = paddle.to_tensor([[1, 3, 2], [4, 0, 5]])
        result = paddle.argmin(x)
        self.assertEqual(result.item(), 4)

    def test_argmin_axis(self):
        """测试指定轴 argmin
        Test argmin with axis"""
        x = paddle.to_tensor([[1, 3, 2], [4, 0, 5]])
        result = paddle.argmin(x, axis=1)
        np.testing.assert_array_equal(result.numpy(), np.array([0, 1]))


class TestArgsort(unittest.TestCase):
    """测试 argsort 函数
    Test argsort function"""

    def test_argsort_ascending(self):
        """测试升序 argsort
        Test ascending argsort"""
        x = paddle.to_tensor([3, 1, 2])
        result = paddle.argsort(x)
        np.testing.assert_array_equal(result.numpy(), np.array([1, 2, 0]))

    def test_argsort_descending(self):
        """测试降序 argsort
        Test descending argsort"""
        x = paddle.to_tensor([3, 1, 2])
        result = paddle.argsort(x, descending=True)
        np.testing.assert_array_equal(result.numpy(), np.array([0, 2, 1]))

    def test_argsort_axis(self):
        """测试指定轴 argsort
        Test argsort with axis"""
        x = paddle.randn([3, 4])
        result = paddle.argsort(x, axis=0)
        self.assertEqual(result.shape, [3, 4])


class TestTopk(unittest.TestCase):
    """测试 topk 函数
    Test topk function"""

    def test_topk_basic(self):
        """测试基本 topk
        Test basic topk"""
        x = paddle.to_tensor([3, 1, 4, 1, 5])
        values, indices = paddle.topk(x, k=2)
        np.testing.assert_array_equal(values.numpy(), np.array([5, 4]))

    def test_topk_axis(self):
        """测试指定轴 topk
        Test topk with axis"""
        x = paddle.randn([3, 4])
        values, indices = paddle.topk(x, k=2, axis=1)
        self.assertEqual(values.shape, [3, 2])

    def test_topk_largest_false(self):
        """测试最小 topk
        Test smallest topk"""
        x = paddle.to_tensor([3, 1, 4, 1, 5])
        values, indices = paddle.topk(x, k=2, largest=False)
        np.testing.assert_array_equal(values.numpy(), np.array([1, 1]))


class TestWhere(unittest.TestCase):
    """测试 where 函数
    Test where function"""

    def test_where_basic(self):
        """测试基本 where
        Test basic where"""
        x = paddle.to_tensor([1.0, -2.0, 3.0, -4.0])
        result = paddle.where(x > 0, x, paddle.zeros_like(x))
        expected = np.array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_where_condition_only(self):
        """测试仅条件 where（返回非零索引）
        Test where with condition only"""
        x = paddle.to_tensor([1, 0, 3, 0, 5])
        result = paddle.nonzero(x)
        self.assertIsNotNone(result)


class TestIndexSelect(unittest.TestCase):
    """测试 index_select 函数
    Test index_select function"""

    def test_index_select_basic(self):
        """测试基本 index_select
        Test basic index_select"""
        x = paddle.randn([3, 4, 5])
        index = paddle.to_tensor([0, 2])
        result = paddle.index_select(x, index, axis=1)
        self.assertEqual(result.shape, [3, 2, 5])

    def test_index_select_axis0(self):
        """测试 axis=0 的 index_select
        Test index_select on axis=0"""
        x = paddle.randn([5, 4])
        index = paddle.to_tensor([1, 3])
        result = paddle.index_select(x, index, axis=0)
        self.assertEqual(result.shape, [2, 4])


class TestMaskedSelect(unittest.TestCase):
    """测试 masked_select 函数
    Test masked_select function"""

    def test_masked_select_basic(self):
        """测试基本 masked_select
        Test basic masked_select"""
        x = paddle.to_tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        mask = x > 0
        result = paddle.masked_select(x, mask)
        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


class TestNonzero(unittest.TestCase):
    """测试 nonzero 函数
    Test nonzero function"""

    def test_nonzero_basic(self):
        """测试基本 nonzero
        Test basic nonzero"""
        x = paddle.to_tensor([[0, 1], [2, 0]])
        result = paddle.nonzero(x)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_nonzero_1d(self):
        """测试一维 nonzero
        Test 1D nonzero"""
        x = paddle.to_tensor([0, 1, 0, 2, 0])
        result = paddle.nonzero(x)
        expected = np.array([[1], [3]])
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == '__main__':
    unittest.main()
