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

# [AUTO-GENERATED] Do not edit manually.
# Target source: paddle/phi/kernels/cpu/unique_kernel.cc
# Generated for exercising C++ CPU kernel: UniqueKernel, UniqueRawKernel
#
# 测试 Unique CPU 内核
# Tests for Unique CPU kernel

import unittest

import numpy as np

import paddle


class TestUniqueKernelBasic(unittest.TestCase):
    """基本 Unique 测试 / Basic Unique tests"""

    def setUp(self):
        """初始化测试环境 / Setup test environment"""
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        """清理测试环境 / Teardown test environment"""
        paddle.enable_static()

    def test_unique_sorted_1d(self):
        """测试一维排序 Unique
        Test 1D sorted unique
        """
        x = paddle.to_tensor([3, 1, 2, 1, 3, 4, 2], dtype="int64")
        out = paddle.unique(x)
        expected = np.array([1, 2, 3, 4], dtype="int64")
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_unique_return_index(self):
        """测试返回首次出现索引
        Test unique with return_index=True
        """
        x = paddle.to_tensor([3, 1, 2, 1, 3], dtype="int64")
        out, indices = paddle.unique(x, return_index=True)
        expected_vals = np.array([1, 2, 3], dtype="int64")
        expected_idx = np.array([1, 2, 0], dtype="int64")
        np.testing.assert_array_equal(out.numpy(), expected_vals)
        np.testing.assert_array_equal(indices.numpy(), expected_idx)

    def test_unique_return_inverse(self):
        """测试返回逆向映射
        Test unique with return_inverse=True
        """
        x = paddle.to_tensor([3, 1, 2, 1, 3], dtype="int64")
        out, inverse = paddle.unique(x, return_inverse=True)
        expected_vals = np.array([1, 2, 3], dtype="int64")
        # x[0]=3 -> index 2, x[1]=1 -> index 0, x[2]=2 -> index 1, etc.
        expected_inv = np.array([2, 0, 1, 0, 2], dtype="int64")
        np.testing.assert_array_equal(out.numpy(), expected_vals)
        np.testing.assert_array_equal(inverse.numpy(), expected_inv)

    def test_unique_return_counts(self):
        """测试返回元素计数
        Test unique with return_counts=True
        """
        x = paddle.to_tensor([3, 1, 2, 1, 3, 3], dtype="int64")
        out, counts = paddle.unique(x, return_counts=True)
        expected_vals = np.array([1, 2, 3], dtype="int64")
        expected_counts = np.array([2, 1, 3], dtype="int64")
        np.testing.assert_array_equal(out.numpy(), expected_vals)
        np.testing.assert_array_equal(counts.numpy(), expected_counts)

    def test_unique_all_return(self):
        """测试同时返回所有可选输出
        Test unique with all return options
        """
        x = paddle.to_tensor([4, 2, 1, 2, 4, 1, 3], dtype="int64")
        out, indices, inverse, counts = paddle.unique(
            x, return_index=True, return_inverse=True, return_counts=True
        )
        self.assertEqual(len(out.numpy()), 4)

    def test_unique_all_same(self):
        """测试所有元素相同的 Unique
        Test unique when all elements are the same
        """
        x = paddle.to_tensor([5, 5, 5, 5], dtype="int64")
        out, counts = paddle.unique(x, return_counts=True)
        np.testing.assert_array_equal(out.numpy(), [5])
        np.testing.assert_array_equal(counts.numpy(), [4])

    def test_unique_single_element(self):
        """测试单元素张量的 Unique
        Test unique with single element
        """
        x = paddle.to_tensor([42], dtype="int64")
        out = paddle.unique(x)
        np.testing.assert_array_equal(out.numpy(), [42])

    def test_unique_empty(self):
        """测试空张量的 Unique
        Test unique with empty tensor
        """
        x = paddle.to_tensor([], dtype="int64")
        out = paddle.unique(x)
        np.testing.assert_array_equal(out.numpy(), [])


class TestUniqueKernelDtype(unittest.TestCase):
    """Unique 不同数据类型测试 / Unique tests with different dtypes"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_unique_float32(self):
        """测试 float32 类型的 Unique
        Test unique with float32
        """
        x = paddle.to_tensor([1.5, 2.0, 1.5, 3.0, 2.0], dtype="float32")
        out = paddle.unique(x)
        expected = np.array([1.5, 2.0, 3.0], dtype="float32")
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)

    def test_unique_float64(self):
        """测试 float64 类型的 Unique
        Test unique with float64
        """
        x = paddle.to_tensor([1.1, 2.2, 1.1], dtype="float64")
        out = paddle.unique(x)
        expected = np.array([1.1, 2.2], dtype="float64")
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-10)

    def test_unique_int32(self):
        """测试 int32 类型的 Unique
        Test unique with int32
        """
        x = paddle.to_tensor([1, 3, 2, 3, 1], dtype="int32")
        out = paddle.unique(x)
        expected = np.array([1, 2, 3], dtype="int32")
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_unique_int64(self):
        """测试 int64 类型的 Unique
        Test unique with int64
        """
        x = paddle.to_tensor([10, 20, 10, 30], dtype="int64")
        out = paddle.unique(x)
        expected = np.array([10, 20, 30], dtype="int64")
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_unique_index_dtype_int32(self):
        """测试 int32 索引类型的 Unique
        Test unique with int32 index dtype
        """
        x = paddle.to_tensor([3, 1, 2, 1], dtype="int64")
        out, indices = paddle.unique(x, return_index=True, dtype="int32")
        self.assertEqual(indices.dtype, paddle.int32)


class TestUniqueKernelWithAxis(unittest.TestCase):
    """带 axis 参数的 Unique 测试 / Unique tests with axis parameter"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_unique_axis_0(self):
        """测试沿 axis=0 的 Unique
        Test unique along axis=0
        """
        x = paddle.to_tensor([[1, 2], [3, 4], [1, 2], [5, 6]], dtype="int64")
        out, indices = paddle.unique(x, axis=0, return_index=True)
        # Unique rows: [1,2], [3,4], [5,6]
        expected = np.array([[1, 2], [3, 4], [5, 6]], dtype="int64")
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(indices.numpy(), [0, 1, 3])

    def test_unique_axis_1(self):
        """测试沿 axis=1 的 Unique
        Test unique along axis=1
        """
        x = paddle.to_tensor([[1, 2, 1, 3], [4, 5, 4, 6]], dtype="int64")
        out = paddle.unique(x, axis=1)
        # Unique columns: [1,2,3] and [4,5,6]
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)

    def test_unique_axis_negative(self):
        """测试负 axis 值
        Test unique with negative axis value
        """
        x = paddle.to_tensor([[1, 2, 1], [3, 4, 3]], dtype="int64")
        out_pos = paddle.unique(x, axis=1)
        out_neg = paddle.unique(x, axis=-1)
        np.testing.assert_array_equal(out_pos.numpy(), out_neg.numpy())

    def test_unique_axis_2d_row(self):
        """测试 2D 张量按行 Unique
        Test 2D tensor unique by rows
        """
        x = paddle.to_tensor([[1, 1], [2, 2], [1, 1]], dtype="int64")
        out, counts = paddle.unique(x, axis=0, return_counts=True)
        expected = np.array([[1, 1], [2, 2]], dtype="int64")
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(counts.numpy(), [2, 1])


class TestUniqueKernelEdgeCases(unittest.TestCase):
    """Unique 边界情况测试 / Unique edge case tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_unique_already_sorted(self):
        """测试已排序输入的 Unique
        Test unique with already sorted input
        """
        x = paddle.to_tensor([1, 2, 3, 4, 5], dtype="int64")
        out = paddle.unique(x)
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3, 4, 5])

    def test_unique_reverse_sorted(self):
        """测试逆序输入的 Unique
        Test unique with reverse sorted input
        """
        x = paddle.to_tensor([5, 4, 3, 2, 1], dtype="int64")
        out = paddle.unique(x)
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3, 4, 5])

    def test_unique_large_tensor(self):
        """测试大张量的 Unique
        Test unique with large tensor
        """
        np.random.seed(42)
        data = np.random.randint(0, 100, size=10000, dtype="int64")
        x = paddle.to_tensor(data)
        out, counts = paddle.unique(x, return_counts=True)
        # Total count should match original size
        self.assertEqual(counts.sum().item(), 10000)

    def test_unique_negative_numbers(self):
        """测试包含负数的 Unique
        Test unique with negative numbers
        """
        x = paddle.to_tensor([-3, -1, -2, -1, -3, 0], dtype="int64")
        out = paddle.unique(x)
        expected = np.array([-3, -2, -1, 0], dtype="int64")
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_unique_unsorted_flag(self):
        """测试 sorted=False 参数（应不影响结果，与 PyTorch 兼容）
        Test sorted=False flag (should not affect result, PyTorch compatible)
        """
        x = paddle.to_tensor([3, 1, 2, 1], dtype="int64")
        out_sorted = paddle.unique(x, sorted=True)
        out_unsorted = paddle.unique(x, sorted=False)
        # Both should produce the same sorted result
        np.testing.assert_array_equal(out_sorted.numpy(), out_unsorted.numpy())


if __name__ == "__main__":
    unittest.main()
