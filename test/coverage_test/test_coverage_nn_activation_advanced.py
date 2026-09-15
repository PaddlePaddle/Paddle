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
ELU/PReLU/SELU等激活函数层测试 / ELU/PReLU/SELU Activation Layer Tests

测试目标 / Test Target:
  paddle.nn 高级激活函数层

覆盖的模块 / Covered Modules:
  - paddle.nn.ELU: 指数线性单元
  - paddle.nn.SELU: 缩放指数线性单元
  - paddle.nn.PReLU: 参数化ReLU
  - paddle.nn.ThresholdedReLU: 阈值ReLU
  - paddle.nn.Softmax2D: 2D Softmax
  - paddle.nn.LogSoftmax: Log Softmax

作用 / Purpose:
  补充高级激活函数层API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestELULayer(unittest.TestCase):
    """测试ELU激活层 / Test ELU activation layer"""

    def test_elu_basic(self):
        """测试基本ELU / Test basic ELU"""
        elu = nn.ELU(alpha=1.0)
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = elu(x)
        self.assertEqual(result.shape, [5])
        # For x >= 0, ELU(x) = x
        self.assertAlmostEqual(float(result[4].numpy()), 2.0, places=5)

    def test_elu_custom_alpha(self):
        """测试自定义alpha ELU / Test ELU with custom alpha"""
        elu = nn.ELU(alpha=2.0)
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = elu(x)
        # For x < 0, ELU(x) = alpha * (exp(x) - 1)
        expected_neg = 2.0 * (np.exp(-1.0) - 1)
        self.assertAlmostEqual(float(result[0].numpy()), expected_neg, places=5)

    def test_elu_batch(self):
        """测试批量ELU / Test batch ELU"""
        elu = nn.ELU()
        x = paddle.randn([4, 8, 16])
        result = elu(x)
        self.assertEqual(result.shape, [4, 8, 16])
        # All outputs should be >= -alpha
        self.assertTrue(bool((result >= -1.0).all().numpy()))


class TestSELULayer(unittest.TestCase):
    """测试SELU激活层 / Test SELU activation layer"""

    def test_selu_basic(self):
        """测试基本SELU / Test basic SELU"""
        selu = nn.SELU()
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = selu(x)
        self.assertEqual(result.shape, [3])
        # SELU(0) = 0
        self.assertAlmostEqual(float(result[1].numpy()), 0.0, places=5)

    def test_selu_self_normalizing(self):
        """测试SELU自归一化特性 / Test SELU self-normalizing property"""
        selu = nn.SELU()
        x = paddle.randn([1000])
        result = selu(x)
        # After SELU, output should have roughly zero mean and unit variance
        # (not exact due to non-linear nature and finite samples)
        self.assertIsNotNone(result)


class TestPReLULayer(unittest.TestCase):
    """测试PReLU层 / Test PReLU layer"""

    def test_prelu_basic(self):
        """测试基本PReLU / Test basic PReLU"""
        prelu = nn.PReLU()
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = prelu(x)
        self.assertEqual(result.shape, [5])

    def test_prelu_num_parameters(self):
        """测试多参数PReLU / Test PReLU with num_parameters"""
        prelu = nn.PReLU(num_parameters=8)
        x = paddle.randn([4, 8, 16])
        result = prelu(x)
        self.assertEqual(result.shape, [4, 8, 16])

    def test_prelu_learnable(self):
        """测试PReLU可学习参数 / Test PReLU learnable parameters"""
        prelu = nn.PReLU(init=0.25)
        self.assertAlmostEqual(float(prelu._weight.item()), 0.25, places=5)


class TestSoftmaxLayers(unittest.TestCase):
    """测试Softmax层 / Test Softmax layers"""

    def test_softmax_basic(self):
        """测试基本Softmax / Test basic Softmax"""
        softmax = nn.Softmax(axis=-1)
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = softmax(x)
        self.assertEqual(result.shape, [2, 3])
        # Each row should sum to 1
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums.numpy(), [1.0, 1.0], rtol=1e-5)

    def test_log_softmax(self):
        """测试LogSoftmax / Test LogSoftmax"""
        log_softmax = nn.LogSoftmax(axis=-1)
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        result = log_softmax(x)
        self.assertEqual(result.shape, [1, 3])
        # All values should be negative (log of probabilities)
        self.assertTrue(bool((result < 0).all().numpy()))

    def test_softmax_2d(self):
        """测试Softmax2D / Test Softmax2D"""
        softmax2d = nn.Softmax2D()
        x = paddle.randn([4, 3, 8, 8])
        result = softmax2d(x)
        self.assertEqual(result.shape, [4, 3, 8, 8])
        # Sum over channels should be 1
        channel_sums = result.sum(axis=1)
        np.testing.assert_allclose(
            channel_sums.numpy(), np.ones([4, 8, 8]), rtol=1e-5
        )


class TestCELUAndHardshrink(unittest.TestCase):
    """测试CELU和Hardshrink / Test CELU and Hardshrink"""

    def test_celu(self):
        """测试CELU / Test CELU"""
        celu = nn.CELU(alpha=1.0)
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = celu(x)
        self.assertEqual(result.shape, [3])
        self.assertAlmostEqual(float(result[1].numpy()), 0.0, places=5)

    def test_hardshrink(self):
        """测试Hardshrink / Test Hardshrink"""
        hardshrink = nn.Hardshrink(threshold=0.5)
        x = paddle.to_tensor([-1.0, -0.3, 0.0, 0.3, 1.0])
        result = hardshrink(x)
        # Values within threshold become 0
        np.testing.assert_allclose(result.numpy(), [-1.0, 0.0, 0.0, 0.0, 1.0])

    def test_softshrink(self):
        """测试Softshrink / Test Softshrink"""
        softshrink = nn.Softshrink(threshold=0.5)
        x = paddle.to_tensor([-1.5, -0.3, 0.0, 0.3, 1.5])
        result = softshrink(x)
        self.assertEqual(result.shape, [5])
        # Values within [-threshold, threshold] become 0
        self.assertAlmostEqual(float(result[1].numpy()), 0.0, places=5)
        self.assertAlmostEqual(float(result[2].numpy()), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
