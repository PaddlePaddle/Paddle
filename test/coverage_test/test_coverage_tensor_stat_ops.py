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

"""
张量统计操作测试 / Tensor Statistics Operations Tests

测试目标 / Test Target:
  paddle.tensor 统计函数

覆盖的模块 / Covered Modules:
  - paddle.mean/sum/var/std: 统计量
  - paddle.median/nanmedian: 中位数
  - paddle.quantile/nanquantile: 分位数
  - paddle.histc/histogram: 直方图
  - paddle.bincount: 二进制计数
  - paddle.topk: top-k

作用 / Purpose:
  补充统计操作API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestBasicStatistics(unittest.TestCase):
    """测试基本统计函数 / Test basic statistical functions"""

    def test_mean_axis(self):
        """测试沿轴求均值 / Test mean along axis"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = paddle.mean(x, axis=0)
        np.testing.assert_allclose(result.numpy(), [2.5, 3.5, 4.5])

    def test_sum_axis(self):
        """测试沿轴求和 / Test sum along axis"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = paddle.sum(x, axis=1)
        np.testing.assert_allclose(result.numpy(), [6.0, 15.0])

    def test_var(self):
        """测试方差 / Test variance"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.var(x)
        # Paddle var uses unbiased=True by default (ddof=1)
        np.testing.assert_allclose(float(result.numpy()), 2.5, rtol=1e-5)

    def test_std(self):
        """测试标准差 / Test standard deviation"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.std(x)
        self.assertAlmostEqual(
            float(result.numpy()), np.std([1, 2, 3, 4, 5], ddof=1), places=4
        )

    def test_prod(self):
        """测试乘积 / Test product"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        result = paddle.prod(x)
        self.assertAlmostEqual(float(result.numpy()), 24.0, places=5)

    def test_cumsum(self):
        """测试累积和 / Test cumulative sum"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        result = paddle.cumsum(x)
        np.testing.assert_allclose(result.numpy(), [1.0, 3.0, 6.0, 10.0])


class TestMedianAndQuantile(unittest.TestCase):
    """测试中位数和分位数 / Test median and quantile"""

    def test_median_1d(self):
        """测试1D中位数 / Test 1D median"""
        x = paddle.to_tensor([1.0, 3.0, 2.0, 5.0, 4.0])
        result = paddle.median(x)
        self.assertAlmostEqual(float(result.numpy()), 3.0, places=5)

    def test_median_axis(self):
        """测试沿轴中位数 / Test median along axis"""
        x = paddle.to_tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        result = paddle.median(x, axis=1)
        np.testing.assert_allclose(result.numpy(), [3.0, 4.0])

    def test_quantile(self):
        """测试分位数 / Test quantile"""
        x = paddle.to_tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        )
        q50 = paddle.quantile(x, 0.5)
        self.assertAlmostEqual(float(q50.numpy()), 5.5, places=4)

    def test_quantile_multiple(self):
        """测试多分位数 / Test multiple quantiles"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.quantile(x, [0.25, 0.75])
        self.assertEqual(result.shape, [2])


class TestTopKAndSort(unittest.TestCase):
    """测试top-k和排序 / Test top-k and sorting"""

    def test_topk(self):
        """测试top-k / Test top-k"""
        x = paddle.to_tensor([5.0, 1.0, 3.0, 2.0, 4.0])
        values, indices = paddle.topk(x, k=3)
        np.testing.assert_allclose(values.numpy(), [5.0, 4.0, 3.0])
        self.assertEqual(indices.shape, [3])

    def test_topk_smallest(self):
        """测试最小k个值 / Test smallest k values"""
        x = paddle.to_tensor([5.0, 1.0, 3.0, 2.0, 4.0])
        values, indices = paddle.topk(x, k=2, largest=False)
        np.testing.assert_allclose(values.numpy(), [1.0, 2.0])

    def test_sort(self):
        """测试排序 / Test sort"""
        x = paddle.to_tensor([5.0, 1.0, 3.0, 2.0, 4.0])
        result = paddle.sort(x)
        np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_argsort(self):
        """测试排序索引 / Test sort indices"""
        x = paddle.to_tensor([3.0, 1.0, 2.0])
        result = paddle.argsort(x)
        np.testing.assert_allclose(result.numpy(), [1, 2, 0])

    def test_sort_descending(self):
        """测试降序排序 / Test descending sort"""
        x = paddle.to_tensor([1.0, 3.0, 2.0])
        result = paddle.sort(x, descending=True)
        np.testing.assert_allclose(result.numpy(), [3.0, 2.0, 1.0])


class TestHistogramAndBincount(unittest.TestCase):
    """测试直方图和计数 / Test histogram and bincount"""

    def test_histc(self):
        """测试histogram直方图 / Test histogram"""
        x = paddle.to_tensor([1.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        result = paddle.histogram(x, bins=3, min=1.0, max=3.0)
        self.assertEqual(result.shape, [3])
        # 3 ones, 2 twos, 1 three
        np.testing.assert_allclose(result.numpy(), [3, 2, 1])

    def test_bincount(self):
        """测试bincount / Test bincount"""
        x = paddle.to_tensor([0, 1, 1, 2, 2, 2])
        result = paddle.bincount(x)
        np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0])


class TestCountNonzero(unittest.TestCase):
    """测试非零元素计数 / Test nonzero counting"""

    def test_count_nonzero(self):
        """测试非零元素数量 / Test count of nonzero elements"""
        x = paddle.to_tensor([0.0, 1.0, 0.0, 2.0, 3.0])
        result = paddle.count_nonzero(x)
        self.assertEqual(int(result.numpy()), 3)

    def test_count_nonzero_axis(self):
        """测试沿轴的非零元素数量 / Test nonzero count along axis"""
        x = paddle.to_tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 0.0]])
        result = paddle.count_nonzero(x, axis=1)
        np.testing.assert_allclose(result.numpy(), [2, 1])

    def test_any_all(self):
        """测试any/all逻辑操作 / Test any/all logic operations"""
        x = paddle.to_tensor([True, False, True])
        self.assertTrue(bool(paddle.any(x).numpy()))
        self.assertFalse(bool(paddle.all(x).numpy()))

        y = paddle.to_tensor([True, True, True])
        self.assertTrue(bool(paddle.all(y).numpy()))


if __name__ == '__main__':
    unittest.main()
