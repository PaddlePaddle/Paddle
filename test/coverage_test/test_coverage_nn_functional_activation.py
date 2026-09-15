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

# [AUTO-GENERATED] Test file for paddle.nn.functional.activation
# 覆盖模块: paddle/nn/functional/activation.py
# 未覆盖行: 98,101,102,103,109,152,155,156,157,163,221,224,225,226,232,277,281,282,283,289,363,367,368,369,375,417,431,432,433,434
# Covered module: paddle/nn/functional/activation.py
# Uncovered lines: 98,101,102,103,109,152,155,156,157,163,221,224,225,226,232,277,281,282,283,289,363,367,368,369,375,417,431,432,433,434

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestSilu(unittest.TestCase):
    """测试 silu 激活函数
    Test silu activation function"""

    def test_silu_basic(self):
        """测试基本的 silu
        Test basic silu"""
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = F.silu(x)
        # silu(x) = x * sigmoid(x)
        expected = x.numpy() * (1.0 / (1.0 + np.exp(-x.numpy())))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_silu_2d(self):
        """测试2D输入的 silu
        Test silu with 2D input"""
        x = paddle.randn([3, 4])
        result = F.silu(x)
        self.assertEqual(result.shape, [3, 4])


class TestLogSigmoid(unittest.TestCase):
    """测试 log_sigmoid 激活函数
    Test log_sigmoid activation function"""

    def test_log_sigmoid_basic(self):
        """测试基本的 log_sigmoid
        Test basic log_sigmoid"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = F.log_sigmoid(x)
        expected = np.log(1.0 / (1.0 + np.exp(-x.numpy())))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_log_sigmoid_2d(self):
        """测试2D输入的 log_sigmoid
        Test log_sigmoid with 2D input"""
        x = paddle.randn([2, 3])
        result = F.log_sigmoid(x)
        self.assertEqual(result.shape, [2, 3])


class TestTanhshrink(unittest.TestCase):
    """测试 tanhshrink 激活函数
    Test tanhshrink activation function"""

    def test_tanhshrink_basic(self):
        """测试基本的 tanhshrink
        Test basic tanhshrink"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = F.tanhshrink(x)
        # tanhshrink(x) = x - tanh(x)
        expected = x.numpy() - np.tanh(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_tanhshrink_2d(self):
        """测试2D输入的 tanhshrink
        Test tanhshrink with 2D input"""
        x = paddle.randn([2, 3])
        result = F.tanhshrink(x)
        self.assertEqual(result.shape, [2, 3])


class TestHardshrink(unittest.TestCase):
    """测试 hardshrink 激活函数
    Test hardshrink activation function"""

    def test_hardshrink_basic(self):
        """测试基本的 hardshrink
        Test basic hardshrink"""
        x = paddle.to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = F.hardshrink(x)
        # hardshrink(x) = x if |x| > 0.5, else 0
        expected = [-2.0, 0.0, 0.0, 0.0, 2.0]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_hardshrink_custom_threshold(self):
        """测试自定义阈值的 hardshrink
        Test hardshrink with custom threshold"""
        x = paddle.to_tensor([-1.0, -0.8, 0.0, 0.8, 1.0])
        result = F.hardshrink(x, threshold=0.9)
        expected = [-1.0, 0.0, 0.0, 0.0, 1.0]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestSoftshrink(unittest.TestCase):
    """测试 softshrink 激活函数
    Test softshrink activation function"""

    def test_softshrink_basic(self):
        """测试基本的 softshrink
        Test basic softshrink"""
        x = paddle.to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = F.softshrink(x)
        # softshrink(x, lambda=0.5) = x - 0.5 if x > 0.5, x + 0.5 if x < -0.5, 0 otherwise
        expected = [-1.5, 0.0, 0.0, 0.0, 1.5]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_softshrink_custom_lambda(self):
        """测试自定义 lambda 的 softshrink
        Test softshrink with custom lambda"""
        x = paddle.to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = F.softshrink(x, threshold=1.0)
        expected = [-1.0, 0.0, 0.0, 0.0, 1.0]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestSoftsign(unittest.TestCase):
    """测试 softsign 激活函数
    Test softsign activation function"""

    def test_softsign_basic(self):
        """测试基本的 softsign
        Test basic softsign"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = F.softsign(x)
        # softsign(x) = x / (1 + |x|)
        expected = [-0.5, 0.0, 0.5]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestHardtanh(unittest.TestCase):
    """测试 hardtanh 激活函数
    Test hardtanh activation function"""

    def test_hardtanh_basic(self):
        """测试基本的 hardtanh
        Test basic hardtanh"""
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = F.hardtanh(x)
        expected = [-1.0, -1.0, 0.0, 1.0, 1.0]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_hardtanh_custom_range(self):
        """测试自定义范围的 hardtanh
        Test hardtanh with custom range"""
        x = paddle.to_tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = F.hardtanh(x, min=-2.0, max=2.0)
        expected = [-2.0, -1.0, 0.0, 1.0, 2.0]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestHardsigmoid(unittest.TestCase):
    """测试 hardsigmoid 激活函数
    Test hardsigmoid activation function"""

    def test_hardsigmoid_basic(self):
        """测试基本的 hardsigmoid
        Test basic hardsigmoid"""
        x = paddle.to_tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
        result = F.hardsigmoid(x)
        # hardsigmoid(x) = clip(x/6 + 0.5, 0, 1)
        expected = [0.0, 1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestHardswish(unittest.TestCase):
    """测试 hardswish 激活函数
    Test hardswish activation function"""

    def test_hardswish_basic(self):
        """测试基本的 hardswish
        Test basic hardswish"""
        x = paddle.to_tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
        result = F.hardswish(x)
        self.assertEqual(result.shape, [5])


class TestPrelu(unittest.TestCase):
    """测试 prelu 激活函数
    Test prelu activation function"""

    def test_prelu_basic(self):
        """测试基本的 prelu
        Test basic prelu"""
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        w = paddle.to_tensor([0.25])
        result = F.prelu(x, w)
        # prelu(x) = max(0, x) + w * min(0, x)
        expected = [0.25 * (-2.0), 0.25 * (-1.0), 0.0, 1.0, 2.0]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestRelu6(unittest.TestCase):
    """测试 relu6 激活函数
    Test relu6 activation function"""

    def test_relu6_basic(self):
        """测试基本的 relu6
        Test basic relu6"""
        x = paddle.to_tensor([-2.0, 0.0, 3.0, 6.0, 8.0])
        result = F.relu6(x)
        expected = [0.0, 0.0, 3.0, 6.0, 6.0]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestSelu(unittest.TestCase):
    """测试 selu 激活函数
    Test selu activation function"""

    def test_selu_basic(self):
        """测试基本的 selu
        Test basic selu"""
        x = paddle.to_tensor([-2.0, 0.0, 2.0])
        result = F.selu(x)
        self.assertEqual(result.shape, [3])


if __name__ == '__main__':
    unittest.main()
