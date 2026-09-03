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
# Target source: paddle/phi/kernels/cpu/gaussian_kernel.cc
# Generated for exercising C++ CPU kernel: GaussianKernel, GaussianInplaceKernel
#
# 测试高斯随机数生成 CPU 内核
# Tests for Gaussian random number generation CPU kernel

import unittest

import numpy as np

import paddle


class TestGaussianKernelBasic(unittest.TestCase):
    """基本高斯分布测试 / Basic Gaussian distribution tests"""

    def setUp(self):
        """初始化测试环境 / Setup test environment"""
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        """清理测试环境 / Teardown test environment"""
        paddle.enable_static()

    def test_gaussian_standard_normal(self):
        """测试标准正态分布 N(0,1)
        Test standard normal distribution N(0,1)
        """
        paddle.seed(42)
        t = paddle.normal(shape=[10000], mean=0.0, std=1.0)
        np.testing.assert_allclose(t.mean().item(), 0.0, atol=0.1)
        self.assertAlmostEqual(t.std().item(), 1.0, delta=0.1)

    def test_gaussian_custom_mean_std(self):
        """测试自定义均值和标准差
        Test custom mean and standard deviation
        """
        paddle.seed(42)
        mean, std = 5.0, 2.0
        t = paddle.normal(shape=[10000], mean=mean, std=std)
        np.testing.assert_allclose(t.mean().item(), mean, atol=0.2)
        self.assertAlmostEqual(t.std().item(), std, delta=0.2)

    def test_gaussian_zero_std(self):
        """测试标准差为零时输出全为均值
        Test that std=0 produces all values equal to mean
        """
        paddle.seed(42)
        mean = 3.14
        t = paddle.normal(shape=[100], mean=mean, std=0.0)
        np.testing.assert_allclose(t.numpy(), np.full(100, mean), atol=1e-6)

    def test_gaussian_shape(self):
        """测试不同形状的高斯分布
        Test Gaussian distribution with different shapes
        """
        for shape in [(10,), (10, 20), (2, 3, 4), (1, 1, 5, 5)]:
            t = paddle.normal(shape=shape)
            self.assertEqual(t.shape, shape)

    def test_gaussian_dtypes(self):
        """测试不同数据类型的高斯分布
        Test Gaussian distribution with different dtypes
        """
        paddle.seed(42)
        # Default dtype is float32
        t_f32 = paddle.normal(shape=[100])
        self.assertEqual(t_f32.dtype, paddle.float32)

        # Use float64 mean to get float64 output
        paddle.seed(42)
        t_f64 = paddle.normal(mean=0.0, std=1.0, shape=[100])
        # Provide a float64 mean tensor to get float64 output
        mean_f64 = paddle.to_tensor(0.0, dtype="float64")
        t_f64 = paddle.normal(mean=mean_f64, std=1.0, shape=[100])
        self.assertEqual(t_f64.dtype, paddle.float64)

    def test_gaussian_fixed_seed_reproducibility(self):
        """测试固定种子产生可重复结果
        Test fixed seed produces reproducible results
        """
        paddle.seed(123)
        t1 = paddle.normal(shape=[100], mean=0.0, std=1.0)

        paddle.seed(123)
        t2 = paddle.normal(shape=[100], mean=0.0, std=1.0)

        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_gaussian_single_element(self):
        """测试单元素高斯分布
        Test single element Gaussian
        """
        t = paddle.normal(shape=[1], mean=0.0, std=1.0)
        self.assertEqual(t.shape, (1,))

    def test_gaussian_negative_mean(self):
        """测试负均值的高斯分布
        Test Gaussian with negative mean
        """
        paddle.seed(42)
        mean = -10.0
        t = paddle.normal(shape=[10000], mean=mean, std=1.0)
        np.testing.assert_allclose(t.mean().item(), mean, atol=0.1)

    def test_gaussian_large_std(self):
        """测试大标准差的高斯分布
        Test Gaussian with large standard deviation
        """
        paddle.seed(42)
        std = 100.0
        t = paddle.normal(shape=[10000], mean=0.0, std=std)
        self.assertAlmostEqual(t.std().item(), std, delta=10.0)

    def test_gaussian_inplace(self):
        """测试原地高斯填充
        Test in-place Gaussian fill via paddle.normal_
        """
        x = paddle.zeros([100], dtype="float32")
        paddle.seed(42)
        x.normal_(mean=0.0, std=1.0)
        self.assertAlmostEqual(x.mean().item(), 0.0, delta=0.3)
        self.assertAlmostEqual(x.std().item(), 1.0, delta=0.3)


class TestGaussianKernelDistributions(unittest.TestCase):
    """高斯分布统计特性测试 / Gaussian distribution statistical tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_gaussian_mean_far_from_data(self):
        """测试高斯分布均值偏离数据
        Test Gaussian distribution with mean far from zero
        """
        paddle.seed(42)
        mean = 1000.0
        std = 10.0
        t = paddle.normal(shape=[50000], mean=mean, std=std)
        np.testing.assert_allclose(t.mean().item(), mean, atol=1.0)
        self.assertAlmostEqual(t.std().item(), std, delta=1.0)

    def test_gaussian_very_small_std(self):
        """测试极小标准差的高斯分布
        Test Gaussian with very small standard deviation
        """
        paddle.seed(42)
        mean = 5.0
        std = 1e-6
        t = paddle.normal(shape=[1000], mean=mean, std=std)
        np.testing.assert_allclose(t.numpy(), np.full(1000, mean), atol=1e-3)

    def test_gaussian_2d_distribution(self):
        """测试二维张量的高斯分布统计特性
        Test Gaussian distribution statistics for 2D tensor
        """
        paddle.seed(42)
        t = paddle.normal(shape=[100, 100], mean=3.0, std=2.0)
        np.testing.assert_allclose(t.mean().item(), 3.0, atol=0.3)
        self.assertAlmostEqual(t.std().item(), 2.0, delta=0.3)

    def test_gaussian_negative_std(self):
        """测试负标准差的高斯分布（与正标准差行为相同）
        Test Gaussian distribution with negative std (same as positive)
        """
        paddle.seed(42)
        t_pos = paddle.normal(shape=[1000], mean=0.0, std=1.0)
        paddle.seed(42)
        t_neg = paddle.normal(shape=[1000], mean=0.0, std=-1.0)
        # Paddle treats negative std as absolute value
        # Both should generate valid random values with the same std
        self.assertAlmostEqual(t_pos.std().item(), 1.0, delta=0.2)
        self.assertAlmostEqual(t_neg.std().item(), 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
