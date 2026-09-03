# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.tensor.search
# 自动生成的单测，覆盖 paddle.tensor.search 模块中未覆盖的代码

"""
测试模块：paddle.tensor.search (argsort, topk, searchsorted, index_select, masked_select)
Test Module: paddle.tensor.search

本测试覆盖以下功能：
This test covers the following functions:
1. argsort - 排序索引 / Argsort with stable/descending options
2. topk - 前k个最大值 / Top-k values
3. searchsorted - 有序搜索 / Sorted search with right parameter
4. index_select - 按索引选择 / Index select with axis
5. masked_select - 按mask选择 / Masked select

覆盖的未覆盖行：argsort stable分支, searchsorted right分支
"""

import unittest

import numpy as np

import paddle


class TestArgsort(unittest.TestCase):
    """测试argsort排序索引
    Test argsort function"""

    def setUp(self):
        paddle.disable_static()

    def test_argsort_ascending(self):
        """升序排序 / Ascending sort"""
        x = paddle.to_tensor([3.0, 1.0, 2.0], dtype='float32')
        out = paddle.argsort(x)
        np.testing.assert_array_equal(out.numpy(), [1, 2, 0])

    def test_argsort_descending(self):
        """降序排序 / Descending sort"""
        x = paddle.to_tensor([3.0, 1.0, 2.0], dtype='float32')
        out = paddle.argsort(x, descending=True)
        np.testing.assert_array_equal(out.numpy(), [0, 2, 1])

    def test_argsort_stable(self):
        """稳定排序 / Stable sort"""
        x = paddle.to_tensor([1.0, 0.0] * 10, dtype='float32')
        out = paddle.argsort(x, stable=True)
        # 稳定排序中相等元素保持原始顺序
        # In stable sort, equal elements maintain original order
        zeros_indices = out.numpy()[:10]
        ones_indices = out.numpy()[10:]
        self.assertTrue(
            all(zeros_indices[i] < zeros_indices[i + 1] for i in range(9))
        )
        self.assertTrue(
            all(ones_indices[i] < ones_indices[i + 1] for i in range(9))
        )

    def test_argsort_2d(self):
        """2D排序 / 2D argsort"""
        x = paddle.to_tensor([[3.0, 1.0], [2.0, 4.0]], dtype='float32')
        out = paddle.argsort(x, axis=1)
        np.testing.assert_array_equal(out.numpy(), [[1, 0], [0, 1]])

    def test_argsort_negative_axis(self):
        """负axis排序 / Argsort with negative axis"""
        x = paddle.to_tensor([[3.0, 1.0], [2.0, 4.0]], dtype='float32')
        out = paddle.argsort(x, axis=-1)
        np.testing.assert_array_equal(out.numpy(), [[1, 0], [0, 1]])


class TestTopk(unittest.TestCase):
    """测试topk前k个最大值
    Test topk function"""

    def setUp(self):
        paddle.disable_static()

    def test_topk_basic(self):
        """基本topk / Basic topk"""
        x = paddle.to_tensor([3.0, 1.0, 5.0, 2.0, 4.0], dtype='float32')
        values, indices = paddle.topk(x, k=3)
        np.testing.assert_allclose(values.numpy(), [5.0, 4.0, 3.0], rtol=1e-5)

    def test_topk_smallest(self):
        """最小的k个 / Smallest k values"""
        x = paddle.to_tensor([3.0, 1.0, 5.0, 2.0, 4.0], dtype='float32')
        values, indices = paddle.topk(x, k=3, largest=False)
        np.testing.assert_allclose(values.numpy(), [1.0, 2.0, 3.0], rtol=1e-5)

    def test_topk_2d(self):
        """2D topk / 2D topk"""
        x = paddle.to_tensor(
            [[3.0, 1.0, 5.0], [4.0, 2.0, 6.0]], dtype='float32'
        )
        values, indices = paddle.topk(x, k=2, axis=-1)
        self.assertEqual(list(values.shape), [2, 2])


class TestSearchsorted(unittest.TestCase):
    """测试searchsorted有序搜索
    Test searchsorted function"""

    def setUp(self):
        paddle.disable_static()

    def test_searchsorted_basic(self):
        """基本searchsorted / Basic searchsorted"""
        sorted_seq = paddle.to_tensor(
            [1.0, 3.0, 5.0, 7.0, 9.0], dtype='float32'
        )
        values = paddle.to_tensor([2.0, 4.0, 6.0], dtype='float32')
        out = paddle.searchsorted(sorted_seq, values)
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3])

    def test_searchsorted_right(self):
        """右侧搜索 / Search sorted with right=True"""
        sorted_seq = paddle.to_tensor([1.0, 3.0, 3.0, 5.0], dtype='float32')
        values = paddle.to_tensor([3.0], dtype='float32')
        out_left = paddle.searchsorted(sorted_seq, values, right=False)
        out_right = paddle.searchsorted(sorted_seq, values, right=True)
        self.assertTrue(int(out_right.numpy()[0]) >= int(out_left.numpy()[0]))


class TestIndexSelect(unittest.TestCase):
    """测试index_select按索引选择
    Test index_select function"""

    def setUp(self):
        paddle.disable_static()

    def test_index_select_axis0(self):
        """axis=0选择 / Index select along axis 0"""
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype='float32'
        )
        index = paddle.to_tensor([0, 2], dtype='int32')
        out = paddle.index_select(x, index, axis=0)
        expected = np.array([[1.0, 2.0], [5.0, 6.0]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_index_select_axis1(self):
        """axis=1选择 / Index select along axis 1"""
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'
        )
        index = paddle.to_tensor([0, 2], dtype='int32')
        out = paddle.index_select(x, index, axis=1)
        expected = np.array([[1.0, 3.0], [4.0, 6.0]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)


class TestMaskedSelect(unittest.TestCase):
    """测试masked_select按mask选择
    Test masked_select function"""

    def setUp(self):
        paddle.disable_static()

    def test_masked_select_basic(self):
        """基本mask选择 / Basic masked select"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype='float32')
        mask = paddle.to_tensor([True, False, True, False])
        out = paddle.masked_select(x, mask)
        np.testing.assert_allclose(out.numpy(), [1.0, 3.0], rtol=1e-5)

    def test_masked_select_2d(self):
        """2D mask选择 / 2D masked select"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        mask = x > 2.0
        out = paddle.masked_select(x, mask)
        np.testing.assert_allclose(out.numpy(), [3.0, 4.0], rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
