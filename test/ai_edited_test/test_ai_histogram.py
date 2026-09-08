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

# [AUTO-GENERATED] Tests for phi/kernels/cpu/histogram_kernel.cc
# histogram_kernel.cc: CPU histogram kernel (bin counting, weighted, density modes)
# Supports float32/float64/int32/int64 input types.

import unittest

import numpy as np

import paddle


class TestHistogramKernel(unittest.TestCase):
    """Test suite for paddle.histogram CPU kernel.

    测试 paddle.histogram 的 CPU 内核，涵盖基本分箱、加权、密度模式、边界情况等场景。
    """

    def setUp(self):
        paddle.set_device('cpu')

    def test_histogram_basic(self):
        """Test basic histogram bin counting.

        测试基本的直方图分箱计数。
        """
        x = paddle.to_tensor([1.0, 2.0, 1.0, 3.0, 2.0, 5.0])
        result = paddle.histogram(x, bins=5, min=0.0, max=6.0)
        # bins: [0,1.2), [1.2,2.4), [2.4,3.6), [3.6,4.8), [4.8,6.0]
        # values: 1,2,1,3,2,5
        # 1 -> bin 0, 2 -> bin 1, 1 -> bin 0, 3 -> bin 2, 2 -> bin 1, 5 -> bin 4
        expected = np.array([2, 2, 1, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_uniform(self):
        """Test histogram with uniformly distributed data.

        测试均匀分布数据的直方图。
        """
        x = paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        result = paddle.histogram(x, bins=6, min=0.0, max=6.0)
        # Each value falls into a unique bin
        expected = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_all_same_bin(self):
        """Test histogram where all values fall in the same bin.

        测试所有值落在同一分箱中的情况。
        """
        x = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
        result = paddle.histogram(x, bins=5, min=0.0, max=5.0)
        # bins: [0,1), [1,2), [2,3), [3,4), [4,5]
        # 1.0 falls in bin 1 (since bin = floor(1.0/5*5) = 1)
        expected = np.array([0, 4, 0, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_boundary_values(self):
        """Test histogram with values at bin boundaries.

        测试值落在分箱边界上的情况。
        """
        x = paddle.to_tensor([0.0, 3.0, 6.0])
        result = paddle.histogram(x, bins=3, min=0.0, max=6.0)
        # bins: [0,2), [2,4), [4,6]
        # 0.0 -> bin 0, 3.0 -> bin 1, 6.0 -> bin 2 (inclusive max)
        expected = np.array([1, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_empty_input(self):
        """Test histogram with empty input tensor.

        测试空输入张量的直方图（应返回全零）。
        """
        x = paddle.to_tensor([], dtype='float32')
        result = paddle.histogram(x, bins=5, min=0.0, max=10.0)
        expected = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_weighted(self):
        """Test histogram with weight tensor.

        测试带权重张量的加权直方图。
        """
        x = paddle.to_tensor([1.0, 2.0, 1.0, 3.0, 2.0, 5.0])
        w = paddle.to_tensor([1.0, 2.0, 1.0, 1.0, 2.0, 1.0])
        result = paddle.histogram(x, bins=5, min=0.0, max=6.0, weight=w)
        # bin 0: 1.0*1 + 1.0*1 = 2.0, bin 1: 2.0*2 + 2.0*2 = 4.0
        # bin 2: 3.0*1 = 1.0, bin 4: 5.0*1 = 1.0
        expected = np.array([2.0, 4.0, 1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_histogram_density(self):
        """Test histogram with density=True.

        测试归一化密度直方图（面积积分为 1）。
        """
        x = paddle.to_tensor([0.5, 1.5, 2.5])
        result = paddle.histogram(x, bins=3, min=0.0, max=3.0, density=True)
        # bins: [0,1), [1,2), [2,3], each has 1 element
        # bin_width = 1.0, total_count = 3
        # density[i] = count[i] / (total * bin_width) = 1/3
        expected = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_histogram_density_integration(self):
        """Test that density histogram integrates to approximately 1.

        测试密度直方图的面积积分约为 1。
        """
        x = paddle.to_tensor([1.0, 2.0, 1.0, 3.0, 2.0, 5.0])
        result = paddle.histogram(x, bins=5, min=0.0, max=6.0, density=True)
        bin_width = (6.0 - 0.0) / 5.0
        integral = float(paddle.sum(result).numpy()) * bin_width
        self.assertAlmostEqual(integral, 1.0, places=4)

    def test_histogram_float64(self):
        """Test histogram with float64 input.

        测试 float64 输入的直方图。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float64')
        result = paddle.histogram(x, bins=3, min=0.0, max=4.0)
        expected = np.array([1, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_int32(self):
        """Test histogram with int32 input.

        测试 int32 输入的直方图。
        """
        x = paddle.to_tensor([1, 2, 1, 3, 2, 5], dtype='int32')
        result = paddle.histogram(x, bins=5, min=0, max=6)
        expected = np.array([2, 2, 1, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_int64(self):
        """Test histogram with int64 input.

        测试 int64 输入的直方图。
        """
        x = paddle.to_tensor([1, 2, 1, 3], dtype='int64')
        result = paddle.histogram(x, bins=3, min=0, max=4)
        expected = np.array([2, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_auto_range(self):
        """Test histogram with min=max=0 (auto range from data).

        测试 min=max=0 时自动从数据范围确定分箱。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.histogram(x, bins=5, min=0.0, max=0.0)
        # When min==max==0, kernel uses data min/max
        # data range: [1, 5], bins=5, width=1
        expected = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_weighted_density(self):
        """Test histogram with both weight and density.

        测试同时使用权重和密度模式的直方图。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        w = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.histogram(
            x, bins=3, min=0.0, max=4.0, weight=w, density=True
        )
        # Each bin has 1 element with respective weight
        # sum of weights = 6, bin_width = 4/3
        # density = weight / (sum_weights * bin_width)
        bin_width = 4.0 / 3.0
        expected_sum = 1.0 + 2.0 + 3.0
        expected = np.array(
            [
                1.0 / (expected_sum * bin_width),
                2.0 / (expected_sum * bin_width),
                3.0 / (expected_sum * bin_width),
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_histogram_2d_input(self):
        """Test histogram with 2D input tensor (flattened internally).

        测试 2D 输入张量的直方图。
        """
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result = paddle.histogram(x, bins=4, min=0.0, max=5.0)
        expected = np.array([1, 1, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_histogram_large_bins(self):
        """Test histogram with many bins.

        测试大分箱数的直方图。
        """
        x = paddle.to_tensor([0.5, 1.5, 2.5, 3.5])
        result = paddle.histogram(x, bins=100, min=0.0, max=4.0)
        self.assertEqual(result.shape[0], 100)
        total = int(paddle.sum(result).numpy())
        self.assertEqual(total, 4)

    def test_histogram_clamped_values(self):
        """Test histogram with values outside [min, max] range (excluded).

        测试值超出 [min, max] 范围时被排除。
        """
        x = paddle.to_tensor([-1.0, 0.5, 2.5, 10.0])
        result = paddle.histogram(x, bins=3, min=0.0, max=3.0)
        # -1.0 and 10.0 are out of range
        expected = np.array([1, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == '__main__':
    unittest.main()
