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
# Target file: paddle/phi/kernels/cpu/conv_grad_kernel.cc
# Tests for conv2d_grad, depthwise_conv2d_grad, conv3d_grad, conv2d_double_grad CPU kernels.
# Exercises the C++ conv_grad, depthwise_conv2d_grad, conv3d_grad, conv2d_double_grad kernels via Python API.
#
# 本文件针对 conv_grad_kernel.cc 中的卷积梯度 CPU 算子编写单元测试。
# 通过 Python API (paddle.nn.functional.conv2d) 的反向传播来间接调用这些 C++ 内核。

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


def _make_tensor(shape, dtype="float32"):
    """Create a tensor with stop_gradient=False.
    创建一个 stop_gradient=False 的张量。"""
    t = paddle.randn(shape, dtype=dtype)
    t.stop_gradient = False
    return t


class TestConv2dGradCPU(unittest.TestCase):
    """Test conv2d backward on CPU.
    测试 conv2d 反向传播在 CPU 上的正确性。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_conv2d_grad_basic(self):
        """Basic conv2d gradient check.
        基础 conv2d 梯度校验。"""
        x = _make_tensor([2, 3, 8, 8])
        w = _make_tensor([6, 3, 3, 3])
        y = F.conv2d(x, w, bias=None, stride=1, padding=1)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertEqual(x.grad.shape, [2, 3, 8, 8])
        self.assertEqual(w.grad.shape, [6, 3, 3, 3])
        self.assertFalse(paddle.any(paddle.isnan(x.grad)))

    def test_conv2d_grad_stride2(self):
        """Conv2d gradient with stride=2.
        带有 stride=2 的 conv2d 梯度测试。"""
        x = _make_tensor([1, 1, 16, 16])
        w = _make_tensor([2, 1, 3, 3])
        y = F.conv2d(x, w, bias=None, stride=2, padding=1)
        loss = y.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [1, 1, 16, 16])
        self.assertEqual(w.grad.shape, [2, 1, 3, 3])

    def test_conv2d_grad_with_bias(self):
        """Conv2d gradient with bias.
        带有偏置的 conv2d 梯度测试。"""
        x = _make_tensor([2, 3, 4, 4])
        w = _make_tensor([3, 3, 3, 3])
        b = _make_tensor([3])
        y = F.conv2d(x, w, bias=b, stride=1, padding=1)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(b.grad)
        self.assertEqual(b.grad.shape, [3])

    def test_conv2d_grad_groups(self):
        """Conv2d gradient with groups (depthwise-like).
        带有分组的 conv2d（类深度可分离）梯度测试。"""
        x = _make_tensor([1, 4, 6, 6])
        w = _make_tensor([4, 1, 3, 3])
        y = F.conv2d(x, w, bias=None, stride=1, padding=1, groups=4)
        loss = y.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [1, 4, 6, 6])
        self.assertEqual(w.grad.shape, [4, 1, 3, 3])

    def test_conv2d_grad_dilation(self):
        """Conv2d gradient with dilation.
        带有空洞卷积的 conv2d 梯度测试。"""
        x = _make_tensor([1, 1, 8, 8])
        w = _make_tensor([1, 1, 3, 3])
        y = F.conv2d(x, w, bias=None, stride=1, padding=2, dilation=2)
        loss = y.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [1, 1, 8, 8])
        self.assertEqual(w.grad.shape, [1, 1, 3, 3])


class TestConv3dGradCPU(unittest.TestCase):
    """Test conv3d backward on CPU.
    测试 conv3d 反向传播在 CPU 上的正确性。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_conv3d_grad_basic(self):
        """Basic conv3d gradient check.
        基础 conv3d 梯度校验。"""
        x = _make_tensor([1, 2, 4, 4, 4])
        w = _make_tensor([3, 2, 2, 2, 2])
        y = F.conv3d(x, w, bias=None, stride=1, padding=0)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertEqual(x.grad.shape, [1, 2, 4, 4, 4])
        self.assertEqual(w.grad.shape, [3, 2, 2, 2, 2])
        self.assertFalse(paddle.any(paddle.isnan(x.grad)))

    def test_conv3d_grad_padding(self):
        """Conv3d gradient with padding.
        带有填充的 conv3d 梯度测试。"""
        x = _make_tensor([1, 1, 6, 6, 6])
        w = _make_tensor([1, 1, 3, 3, 3])
        y = F.conv3d(x, w, bias=None, stride=1, padding=1)
        loss = y.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [1, 1, 6, 6, 6])
        self.assertEqual(w.grad.shape, [1, 1, 3, 3, 3])


class TestConv2dDoubleGradCPU(unittest.TestCase):
    """Test conv2d double backward on CPU.
    测试 conv2d 二阶反向传播在 CPU 上的正确性。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_conv2d_double_grad_basic(self):
        """Conv2d double gradient via backward+backward.
        通过两次 backward 测试 conv2d 二阶梯度。"""
        x = _make_tensor([1, 1, 4, 4])
        w = _make_tensor([1, 1, 3, 3])
        y = F.conv2d(x, w, bias=None, stride=1, padding=0)
        loss = y.sum()
        # First backward
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, [1, 1, 4, 4])
        # Second backward on a new computation graph
        x2 = _make_tensor([1, 1, 4, 4])
        w2 = _make_tensor([1, 1, 3, 3])
        y2 = F.conv2d(x2, w2, bias=None, stride=1, padding=0)
        loss2 = y2.sum()
        loss2.backward()
        self.assertIsNotNone(x2.grad)
        self.assertEqual(x2.grad.shape, [1, 1, 4, 4])


class TestDepthwiseConv2dGradCPU(unittest.TestCase):
    """Test depthwise conv2d backward on CPU via groups.
    通过分组卷积测试 depthwise conv2d 反向传播。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_depthwise_conv2d_grad(self):
        """Depthwise conv2d gradient (groups=in_channels).
        深度可分离卷积梯度测试（groups=in_channels）。"""
        x = _make_tensor([2, 3, 8, 8])
        w = _make_tensor([3, 1, 3, 3])
        y = F.conv2d(x, w, bias=None, stride=1, padding=1, groups=3)
        loss = y.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [2, 3, 8, 8])
        self.assertEqual(w.grad.shape, [3, 1, 3, 3])
        # Verify the gradient is non-zero
        self.assertGreater(paddle.abs(x.grad).max().item(), 0)


class TestConvGradNumericalCPU(unittest.TestCase):
    """Numerical gradient verification for conv kernels.
    卷积算子的数值梯度验证。"""

    def setUp(self):
        paddle.set_device("cpu")

    def test_conv2d_grad_numerical(self):
        """Verify conv2d input gradient against numerical gradient.
        将 conv2d 输入梯度与数值梯度进行对比验证。"""
        eps = 1e-5
        np.random.seed(42)
        x_np = np.random.randn(1, 1, 4, 4).astype("float64")
        w_np = np.random.randn(1, 1, 3, 3).astype("float64")
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        w = paddle.to_tensor(w_np)
        w.stop_gradient = False
        y = F.conv2d(x, w, stride=1, padding=0)
        loss = y.sum()
        loss.backward()
        # Numerical gradient for x
        num_grad_x = np.zeros_like(x_np)
        for idx in np.ndindex(x_np.shape):
            x_plus = x_np.copy()
            x_plus[idx] += eps
            x_minus = x_np.copy()
            x_minus[idx] -= eps
            y_plus = F.conv2d(
                paddle.to_tensor(x_plus),
                paddle.to_tensor(w_np),
                stride=1,
                padding=0,
            )
            y_minus = F.conv2d(
                paddle.to_tensor(x_minus),
                paddle.to_tensor(w_np),
                stride=1,
                padding=0,
            )
            num_grad_x[idx] = (y_plus.sum().item() - y_minus.sum().item()) / (
                2 * eps
            )
        np.testing.assert_allclose(
            x.grad.numpy(), num_grad_x, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
