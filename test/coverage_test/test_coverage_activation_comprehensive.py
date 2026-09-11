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
# Uncovered lines: celu, elu, gelu, glu, gumbel_softmax, mish, maxout,
#   log_softmax, leaky_relu, softplus, softsign, tanhshrink, thresholded_relu

import unittest

import numpy as np

import paddle


class TestCELU(unittest.TestCase):
    """测试 CELU 激活函数
    Test CELU activation function"""

    def test_celu_basic(self):
        """测试基本 CELU
        Test basic CELU"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = paddle.nn.functional.celu(x)
        self.assertEqual(result.shape, [3])

    def test_celu_alpha(self):
        """测试带 alpha 参数的 CELU
        Test CELU with alpha parameter"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = paddle.nn.functional.celu(x, alpha=2.0)
        self.assertEqual(result.shape, [3])

    def test_celu_positive_passthrough(self):
        """测试 CELU 正值直接通过
        Test CELU positive values pass through"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.nn.functional.celu(x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)


class TestELU(unittest.TestCase):
    """测试 ELU 激活函数
    Test ELU activation function"""

    def test_elu_basic(self):
        """测试基本 ELU
        Test basic ELU"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = paddle.nn.functional.elu(x)
        self.assertEqual(result.shape, [3])

    def test_elu_alpha(self):
        """测试带 alpha 参数的 ELU
        Test ELU with alpha parameter"""
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0])
        result = paddle.nn.functional.elu(x, alpha=2.0)
        self.assertEqual(result.shape, [4])

    def test_elu_positive_passthrough(self):
        """测试 ELU 正值直接通过
        Test ELU positive values pass through"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.nn.functional.elu(x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)


class TestGELU(unittest.TestCase):
    """测试 GELU 激活函数
    Test GELU activation function"""

    def test_gelu_basic(self):
        """测试基本 GELU
        Test basic GELU"""
        x = paddle.randn([3, 4])
        result = paddle.nn.functional.gelu(x)
        self.assertEqual(result.shape, [3, 4])

    def test_gelu_approximate(self):
        """测试近似 GELU
        Test approximate GELU"""
        x = paddle.randn([3, 4])
        result = paddle.nn.functional.gelu(x, approximate=True)
        self.assertEqual(result.shape, [3, 4])


class TestGLU(unittest.TestCase):
    """测试 GLU 激活函数
    Test GLU activation function"""

    def test_glu_basic(self):
        """测试基本 GLU
        Test basic GLU"""
        x = paddle.randn([3, 4])
        result = paddle.nn.functional.glu(x)
        self.assertEqual(result.shape, [3, 2])

    def test_glu_axis(self):
        """测试指定轴的 GLU
        Test GLU with axis"""
        x = paddle.randn([3, 4, 6])
        result = paddle.nn.functional.glu(x, axis=2)
        self.assertEqual(result.shape, [3, 4, 3])


class TestGumbelSoftmax(unittest.TestCase):
    """测试 Gumbel-Softmax 函数
    Test Gumbel-Softmax function"""

    def test_gumbel_softmax_basic(self):
        """测试基本 Gumbel-Softmax
        Test basic Gumbel-Softmax"""
        x = paddle.randn([3, 5])
        result = paddle.nn.functional.gumbel_softmax(x)
        self.assertEqual(result.shape, [3, 5])
        # Output should sum to 1 along last dim
        sums = paddle.sum(result, axis=-1)
        np.testing.assert_allclose(sums.numpy(), np.ones(3), atol=1e-5)

    def test_gumbel_softmax_hard(self):
        """测试 hard Gumbel-Softmax
        Test hard Gumbel-Softmax"""
        x = paddle.randn([3, 5])
        result = paddle.nn.functional.gumbel_softmax(x, hard=True)
        self.assertEqual(result.shape, [3, 5])

    def test_gumbel_softmax_temperature(self):
        """测试不同温度的 Gumbel-Softmax
        Test Gumbel-Softmax with different temperature"""
        x = paddle.randn([3, 5])
        result = paddle.nn.functional.gumbel_softmax(x, temperature=0.5)
        self.assertEqual(result.shape, [3, 5])


class TestMish(unittest.TestCase):
    """测试 Mish 激活函数
    Test Mish activation function"""

    def test_mish_basic(self):
        """测试基本 Mish
        Test basic Mish"""
        x = paddle.randn([3, 4])
        result = paddle.nn.functional.mish(x)
        self.assertEqual(result.shape, [3, 4])

    def test_mish_zero(self):
        """测试零输入的 Mish
        Test Mish with zero input"""
        x = paddle.to_tensor([0.0])
        result = paddle.nn.functional.mish(x)
        self.assertAlmostEqual(result.item(), 0.0, places=5)


class TestMaxout(unittest.TestCase):
    """测试 Maxout 激活函数
    Test Maxout activation function"""

    def test_maxout_basic(self):
        """测试基本 Maxout
        Test basic Maxout"""
        x = paddle.randn([2, 4, 3, 3])
        result = paddle.nn.functional.maxout(x, groups=2)
        self.assertEqual(result.shape, [2, 2, 3, 3])

    def test_maxout_groups(self):
        """测试不同 group 数的 Maxout
        Test Maxout with different groups"""
        x = paddle.randn([1, 6, 2, 2])
        result = paddle.nn.functional.maxout(x, groups=3)
        self.assertEqual(result.shape, [1, 2, 2, 2])


class TestLogSoftmax(unittest.TestCase):
    """测试 LogSoftmax 函数
    Test LogSoftmax function"""

    def test_log_softmax_basic(self):
        """测试基本 LogSoftmax
        Test basic LogSoftmax"""
        x = paddle.randn([3, 5])
        result = paddle.nn.functional.log_softmax(x)
        self.assertEqual(result.shape, [3, 5])

    def test_log_softmax_axis(self):
        """测试指定轴的 LogSoftmax
        Test LogSoftmax with axis"""
        x = paddle.randn([3, 5])
        result = paddle.nn.functional.log_softmax(x, axis=0)
        self.assertEqual(result.shape, [3, 5])

    def test_log_softmax_exp_equals_softmax(self):
        """测试 exp(log_softmax) 等于 softmax
        Test exp(log_softmax) equals softmax"""
        x = paddle.randn([3, 5])
        log_result = paddle.nn.functional.log_softmax(x)
        softmax_result = paddle.nn.functional.softmax(x)
        np.testing.assert_allclose(
            paddle.exp(log_result).numpy(), softmax_result.numpy(), atol=1e-5
        )


class TestLeakyReLU(unittest.TestCase):
    """测试 LeakyReLU 激活函数
    Test LeakyReLU activation function"""

    def test_leaky_relu_basic(self):
        """测试基本 LeakyReLU
        Test basic LeakyReLU"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = paddle.nn.functional.leaky_relu(x)
        self.assertEqual(result.shape, [3])

    def test_leaky_relu_negative_slope(self):
        """测试带 negative_slope 的 LeakyReLU
        Test LeakyReLU with negative_slope"""
        x = paddle.to_tensor([-2.0])
        result = paddle.nn.functional.leaky_relu(x, negative_slope=0.5)
        self.assertAlmostEqual(result.item(), -1.0, places=5)

    def test_leaky_relu_positive_passthrough(self):
        """测试 LeakyReLU 正值直接通过
        Test LeakyReLU positive values pass through"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.nn.functional.leaky_relu(x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)


class TestSoftplus(unittest.TestCase):
    """测试 Softplus 激活函数
    Test Softplus activation function"""

    def test_softplus_basic(self):
        """测试基本 Softplus
        Test basic Softplus"""
        x = paddle.randn([3, 4])
        result = paddle.nn.functional.softplus(x)
        self.assertEqual(result.shape, [3, 4])

    def test_softplus_large_positive(self):
        """测试大正值的 Softplus（接近线性）
        Test Softplus with large positive values (near linear)"""
        x = paddle.to_tensor([100.0])
        result = paddle.nn.functional.softplus(x)
        self.assertAlmostEqual(result.item(), 100.0, places=2)


class TestSoftsign(unittest.TestCase):
    """测试 Softsign 激活函数
    Test Softsign activation function"""

    def test_softsign_basic(self):
        """测试基本 Softsign
        Test basic Softsign"""
        x = paddle.randn([3, 4])
        result = paddle.nn.functional.softsign(x)
        self.assertEqual(result.shape, [3, 4])

    def test_softsign_zero(self):
        """测试零输入的 Softsign
        Test Softsign with zero input"""
        x = paddle.to_tensor([0.0])
        result = paddle.nn.functional.softsign(x)
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_softsign_range(self):
        """测试 Softsign 输出范围 (-1, 1)
        Test Softsign output range (-1, 1)"""
        x = paddle.randn([100])
        result = paddle.nn.functional.softsign(x)
        self.assertTrue(paddle.all(result >= -1.0).item())
        self.assertTrue(paddle.all(result <= 1.0).item())


class TestTanhshrink(unittest.TestCase):
    """测试 Tanhshrink 激活函数
    Test Tanhshrink activation function"""

    def test_tanhshrink_basic(self):
        """测试基本 Tanhshrink
        Test basic Tanhshrink"""
        x = paddle.randn([3, 4])
        result = paddle.nn.functional.tanhshrink(x)
        self.assertEqual(result.shape, [3, 4])

    def test_tanhshrink_zero(self):
        """测试零输入的 Tanhshrink
        Test Tanhshrink with zero input"""
        x = paddle.to_tensor([0.0])
        result = paddle.nn.functional.tanhshrink(x)
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_tanhshrink_formula(self):
        """测试 Tanhshrink 公式 x - tanh(x)
        Test Tanhshrink formula x - tanh(x)"""
        x = paddle.randn([3, 4])
        result = paddle.nn.functional.tanhshrink(x)
        expected = x - paddle.tanh(x)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


class TestThresholdedReLU(unittest.TestCase):
    """测试 ThresholdedReLU 激活函数
    Test ThresholdedReLU activation function"""

    def test_thresholded_relu_basic(self):
        """测试基本 ThresholdedReLU
        Test basic ThresholdedReLU"""
        x = paddle.to_tensor([-1.0, 0.0, 1.5, 2.0])
        result = paddle.nn.functional.thresholded_relu(x)
        self.assertEqual(result.shape, [4])

    def test_thresholded_relu_threshold(self):
        """测试带阈值的 ThresholdedReLU
        Test ThresholdedReLU with threshold"""
        x = paddle.to_tensor([-1.0, 0.5, 1.0, 2.0])
        result = paddle.nn.functional.thresholded_relu(x, threshold=1.0)
        # Values <= threshold should be 0, values > threshold pass through
        self.assertAlmostEqual(result[0].item(), 0.0, places=5)
        self.assertAlmostEqual(result[1].item(), 0.0, places=5)
        self.assertAlmostEqual(
            result[2].item(), 0.0, places=5
        )  # 1.0 is not > 1.0
        self.assertAlmostEqual(result[3].item(), 2.0, places=5)


if __name__ == '__main__':
    unittest.main()
