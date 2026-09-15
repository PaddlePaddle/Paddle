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

# [AUTO-GENERATED] Test file for paddle.tensor.search
# 覆盖模块: paddle/tensor/search.py
# 未覆盖行: 165,181,182,185,188,194,283,284,300,307,309,315,405,421,423,427,588,589,592,595,601,727,728,729,730,731,733,734,736,742
# Covered module: paddle/tensor/search.py
# Uncovered lines: 165,181,182,185,188,194,283,284,300,307,309,315,405,421,423,427,588,589,592,595,601,727,728,729,730,731,733,734,736,742

import unittest

import numpy as np

import paddle


class TestArgsort(unittest.TestCase):
    """测试 argsort 函数
    Test argsort function"""

    def test_argsort_basic(self):
        """测试基本的 argsort
        Test basic argsort"""
        x = paddle.to_tensor([3.0, 1.0, 2.0])
        result = paddle.argsort(x)
        np.testing.assert_array_equal(result.numpy(), [1, 2, 0])

    def test_argsort_descending(self):
        """测试降序 argsort
        Test argsort with descending order"""
        x = paddle.to_tensor([3.0, 1.0, 2.0])
        result = paddle.argsort(x, descending=True)
        np.testing.assert_array_equal(result.numpy(), [0, 2, 1])

    def test_argsort_2d(self):
        """测试2D输入的 argsort
        Test argsort with 2D input"""
        x = paddle.to_tensor([[3.0, 1.0], [2.0, 4.0]])
        result = paddle.argsort(x, axis=1)
        self.assertEqual(result.shape, [2, 2])

    def test_argsort_axis0(self):
        """测试 axis=0 的 argsort
        Test argsort with axis=0"""
        x = paddle.to_tensor([[3.0, 1.0], [2.0, 4.0]])
        result = paddle.argsort(x, axis=0)
        self.assertEqual(result.shape, [2, 2])

    def test_argsort_stable(self):
        """测试 stable=True 的 argsort
        Test argsort with stable=True"""
        x = paddle.to_tensor([3.0, 1.0, 2.0])
        result = paddle.argsort(x, stable=True)
        np.testing.assert_array_equal(result.numpy(), [1, 2, 0])

    def test_argsort_alias_input(self):
        """测试使用 input 别名的 argsort
        Test argsort with 'input' alias parameter"""
        x = paddle.to_tensor([3.0, 1.0, 2.0])
        result = paddle.argsort(input=x)
        np.testing.assert_array_equal(result.numpy(), [1, 2, 0])

    def test_argsort_alias_dim(self):
        """测试使用 dim 别名的 argsort
        Test argsort with 'dim' alias parameter"""
        x = paddle.to_tensor([[3.0, 1.0], [2.0, 4.0]])
        result = paddle.argsort(x, dim=0)
        self.assertEqual(result.shape, [2, 2])


class TestArgmax(unittest.TestCase):
    """测试 argmax 函数
    Test argmax function"""

    def test_argmax_basic(self):
        """测试基本的 argmax
        Test basic argmax"""
        x = paddle.to_tensor([1.0, 3.0, 2.0])
        result = paddle.argmax(x)
        self.assertEqual(result.item(), 1)

    def test_argmax_axis(self):
        """测试指定 axis 的 argmax
        Test argmax with specified axis"""
        x = paddle.to_tensor([[1.0, 3.0], [2.0, 4.0]])
        result = paddle.argmax(x, axis=1)
        np.testing.assert_array_equal(result.numpy(), [1, 1])

    def test_argmax_keepdim(self):
        """测试 keepdim=True 的 argmax
        Test argmax with keepdim=True"""
        x = paddle.to_tensor([[1.0, 3.0], [2.0, 4.0]])
        result = paddle.argmax(x, axis=1, keepdim=True)
        self.assertEqual(result.shape, [2, 1])

    def test_argmax_1d(self):
        """测试1D输入的 argmax
        Test argmax with 1D input"""
        x = paddle.to_tensor([5.0, 2.0, 8.0, 1.0])
        result = paddle.argmax(x)
        self.assertEqual(result.item(), 2)


class TestArgmin(unittest.TestCase):
    """测试 argmin 函数
    Test argmin function"""

    def test_argmin_basic(self):
        """测试基本的 argmin
        Test basic argmin"""
        x = paddle.to_tensor([3.0, 1.0, 2.0])
        result = paddle.argmin(x)
        self.assertEqual(result.item(), 1)

    def test_argmin_axis(self):
        """测试指定 axis 的 argmin
        Test argmin with specified axis"""
        x = paddle.to_tensor([[3.0, 1.0], [2.0, 4.0]])
        result = paddle.argmin(x, axis=1)
        np.testing.assert_array_equal(result.numpy(), [1, 0])

    def test_argmin_keepdim(self):
        """测试 keepdim=True 的 argmin
        Test argmin with keepdim=True"""
        x = paddle.to_tensor([[3.0, 1.0], [2.0, 4.0]])
        result = paddle.argmin(x, axis=0, keepdim=True)
        self.assertEqual(result.shape, [1, 2])


class TestTopk(unittest.TestCase):
    """测试 topk 函数
    Test topk function"""

    def test_topk_basic(self):
        """测试基本的 topk
        Test basic topk"""
        x = paddle.to_tensor([1.0, 4.0, 3.0, 2.0])
        values, indices = paddle.topk(x, k=2)
        np.testing.assert_array_equal(values.numpy(), [4.0, 3.0])
        np.testing.assert_array_equal(indices.numpy(), [1, 2])

    def test_topk_2d(self):
        """测试2D输入的 topk
        Test topk with 2D input"""
        x = paddle.to_tensor([[1.0, 4.0, 3.0], [6.0, 2.0, 5.0]])
        values, indices = paddle.topk(x, k=2, axis=1)
        self.assertEqual(values.shape, [2, 2])
        self.assertEqual(indices.shape, [2, 2])

    def test_topk_smallest(self):
        """测试 largest=False 的 topk
        Test topk with largest=False"""
        x = paddle.to_tensor([1.0, 4.0, 3.0, 2.0])
        values, indices = paddle.topk(x, k=2, largest=False)
        np.testing.assert_array_equal(values.numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(indices.numpy(), [0, 3])

    def test_topk_sorted(self):
        """测试 sorted=True 的 topk
        Test topk with sorted=True"""
        x = paddle.to_tensor([1.0, 4.0, 3.0, 2.0])
        values, indices = paddle.topk(x, k=3, sorted=True)
        self.assertEqual(values.shape, [3])


class TestWhere(unittest.TestCase):
    """测试 where 函数
    Test where function"""

    def test_where_basic(self):
        """测试基本的 where
        Test basic where"""
        condition = paddle.to_tensor([True, False, True])
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        result = paddle.where(condition, x, y)
        np.testing.assert_array_equal(result.numpy(), [1.0, 5.0, 3.0])

    def test_where_scalar(self):
        """测试标量 where
        Test where with scalars"""
        condition = paddle.to_tensor([True, False])
        x = paddle.to_tensor([1.0, 2.0])
        y = paddle.to_tensor([3.0, 4.0])
        result = paddle.where(condition, x, y)
        np.testing.assert_array_equal(result.numpy(), [1.0, 4.0])


class TestIndexSelect(unittest.TestCase):
    """测试 index_select 函数
    Test index_select function"""

    def test_index_select_basic(self):
        """测试基本的 index_select
        Test basic index_select"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = paddle.to_tensor([0, 2], dtype='int64')
        result = paddle.index_select(x, index, axis=0)
        self.assertEqual(result.shape, [2, 2])
        np.testing.assert_array_equal(result.numpy(), [[1.0, 2.0], [5.0, 6.0]])

    def test_index_select_axis1(self):
        """测试 axis=1 的 index_select
        Test index_select with axis=1"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        index = paddle.to_tensor([0, 2], dtype='int64')
        result = paddle.index_select(x, index, axis=1)
        self.assertEqual(result.shape, [2, 2])

    def test_index_select_1d(self):
        """测试1D输入的 index_select
        Test index_select with 1D input"""
        x = paddle.to_tensor([10.0, 20.0, 30.0, 40.0])
        index = paddle.to_tensor([1, 3], dtype='int64')
        result = paddle.index_select(x, index, axis=0)
        np.testing.assert_array_equal(result.numpy(), [20.0, 40.0])


class TestMaskedSelect(unittest.TestCase):
    """测试 masked_select 函数
    Test masked_select function"""

    def test_masked_select_basic(self):
        """测试基本的 masked_select
        Test basic masked_select"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        mask = paddle.to_tensor([True, False, True, False])
        result = paddle.masked_select(x, mask)
        np.testing.assert_array_equal(result.numpy(), [1.0, 3.0])

    def test_masked_select_2d(self):
        """测试2D输入的 masked_select
        Test masked_select with 2D input"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = paddle.to_tensor([[True, False], [False, True]])
        result = paddle.masked_select(x, mask)
        np.testing.assert_array_equal(result.numpy(), [1.0, 4.0])


if __name__ == '__main__':
    unittest.main()
