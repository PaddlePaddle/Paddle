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

# [AUTO-GENERATED]
# Target file: paddle/phi/kernels/funcs/segment_pooling.cc
# Tests for segment pooling CPU kernels.
# Exercises the C++ SegmentPoolFunctor via paddle._C_ops.segment_pool.
#
# 本文件针对 segment_pooling.cc 中的分段池化 CPU 算子编写单元测试。
# 通过 paddle._C_ops.segment_pool 调用 C++ SegmentPoolFunctor，
# 验证 SUM、MEAN、MAX、MIN 四种分段池化模式的正确性。

import unittest

import numpy as np

import paddle


class TestSegmentPoolSUMCPU(unittest.TestCase):
    """Test segment pool with SUM on CPU.
    测试 CPU 上的 SUM 分段池化操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_segment_pool_sum_basic(self):
        """Basic SUM segment pool.
        基础 SUM 分段池化测试。"""
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
        segments = paddle.to_tensor([0, 0, 1, 1, 2], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "SUM")
        expected = np.array([[4.0, 6.0], [12.0, 14.0], [9.0, 10.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_segment_pool_sum_single_element_segments(self):
        """SUM segment pool with single-element segments.
        单元素分段的 SUM 分段池化测试。"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        segments = paddle.to_tensor([0, 1, 2], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "SUM")
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_segment_pool_sum_three_in_one_segment(self):
        """SUM segment pool with three rows in one segment.
        三行同属一个分段的 SUM 分段池化测试。"""
        x = paddle.to_tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        segments = paddle.to_tensor([0, 0, 0, 1], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "SUM")
        expected = np.array([[6.0, 6.0], [4.0, 4.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)


class TestSegmentPoolMEANCPU(unittest.TestCase):
    """Test segment pool with MEAN on CPU.
    测试 CPU 上的 MEAN 分段池化操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_segment_pool_mean_basic(self):
        """Basic MEAN segment pool.
        基础 MEAN 分段池化测试。"""
        x = paddle.to_tensor(
            [[2.0, 4.0], [4.0, 8.0], [10.0, 12.0], [14.0, 16.0], [9.0, 10.0]]
        )
        segments = paddle.to_tensor([0, 0, 1, 1, 2], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "MEAN")
        expected = np.array([[3.0, 6.0], [12.0, 14.0], [9.0, 10.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_segment_pool_mean_single(self):
        """MEAN segment pool with single-element segments.
        单元素分段的 MEAN 分段池化测试。"""
        x = paddle.to_tensor([[5.0, 7.0]])
        segments = paddle.to_tensor([0], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "MEAN")
        np.testing.assert_allclose(result.numpy(), [[5.0, 7.0]], rtol=1e-6)


class TestSegmentPoolMAXCPU(unittest.TestCase):
    """Test segment pool with MAX on CPU.
    测试 CPU 上的 MAX 分段池化操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_segment_pool_max_basic(self):
        """Basic MAX segment pool.
        基础 MAX 分段池化测试。"""
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
        segments = paddle.to_tensor([0, 0, 1, 1, 2], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "MAX")
        expected = np.array([[3.0, 4.0], [7.0, 8.0], [9.0, 10.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_segment_pool_max_negative(self):
        """MAX segment pool with negative values.
        含负值的 MAX 分段池化测试。"""
        x = paddle.to_tensor([[-5.0, -1.0], [-3.0, -4.0], [-2.0, -6.0]])
        segments = paddle.to_tensor([0, 0, 1], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "MAX")
        expected = np.array([[-3.0, -1.0], [-2.0, -6.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)


class TestSegmentPoolMINCPU(unittest.TestCase):
    """Test segment pool with MIN on CPU.
    测试 CPU 上的 MIN 分段池化操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_segment_pool_min_basic(self):
        """Basic MIN segment pool.
        基础 MIN 分段池化测试。"""
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
        segments = paddle.to_tensor([0, 0, 1, 1, 2], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "MIN")
        expected = np.array([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_segment_pool_min_negative(self):
        """MIN segment pool with negative values.
        含负值的 MIN 分段池化测试。"""
        x = paddle.to_tensor([[-5.0, -1.0], [-3.0, -4.0], [-2.0, -6.0]])
        segments = paddle.to_tensor([0, 0, 1], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "MIN")
        expected = np.array([[-5.0, -4.0], [-2.0, -6.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)


class TestSegmentPoolInt64SegmentIdsCPU(unittest.TestCase):
    """Test segment pool with int64 segment IDs on CPU.
    测试 CPU 上使用 int64 分段 ID 的分段池化操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_segment_pool_int64_ids(self):
        """Segment pool with int64 segment IDs.
        使用 int64 分段 ID 的分段池化测试。"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        segments = paddle.to_tensor([0, 0, 1, 1], dtype="int64")
        result = paddle._C_ops.segment_pool(x, segments, "SUM")
        expected = np.array([[4.0, 6.0], [12.0, 14.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_segment_pool_int64_gap_ids(self):
        """Segment pool with non-contiguous int64 segment IDs.
        非连续 int64 分段 ID 的分段池化测试。"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        segments = paddle.to_tensor([0, 0, 5], dtype="int64")
        result = paddle._C_ops.segment_pool(x, segments, "SUM")
        self.assertEqual(result.shape[0], 6)


class TestSegmentPoolOutputShapeCPU(unittest.TestCase):
    """Test segment pool output shapes on CPU.
    测试 CPU 上分段池化的输出形状。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_segment_pool_output_rows(self):
        """Number of output rows = max_segment_id + 1.
        输出行数 = 最大分段 ID + 1。"""
        x = paddle.randn([10, 8])
        segments = paddle.to_tensor(
            [0, 0, 1, 1, 1, 2, 2, 3, 3, 3], dtype="int32"
        )
        for pooltype in ["SUM", "MEAN", "MAX", "MIN"]:
            result = paddle._C_ops.segment_pool(x, segments, pooltype)
            self.assertEqual(result.shape, [4, 8], f"Failed for {pooltype}")

    def test_segment_pool_output_columns(self):
        """Output columns should match input columns.
        输出列数应与输入列数一致。"""
        x = paddle.randn([5, 16])
        segments = paddle.to_tensor([0, 0, 1, 1, 2], dtype="int32")
        result = paddle._C_ops.segment_pool(x, segments, "SUM")
        self.assertEqual(result.shape[1], 16)


if __name__ == "__main__":
    unittest.main()
