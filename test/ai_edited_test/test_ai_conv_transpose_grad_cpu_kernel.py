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
# Target source: paddle/phi/kernels/cpu/conv_transpose_grad_kernel.cc
# Generated for exercising C++ CPU kernel: Conv2dTransposeGradKernel,
#   Conv3dTransposeGradKernel, DepthwiseConv2dTransposeGradKernel
#
# 测试转置卷积梯度 CPU 内核
# Tests for Conv Transpose gradient CPU kernels

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestConv2dTransposeGradKernel(unittest.TestCase):
    """2D 转置卷积梯度内核测试 / 2D Conv Transpose gradient kernel tests"""

    def setUp(self):
        """初始化测试环境 / Setup test environment"""
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        """清理测试环境 / Teardown test environment"""
        paddle.enable_static()

    def test_conv2d_transpose_grad_basic(self):
        """测试基本 2D 转置卷积梯度
        Test basic 2D conv transpose gradient
        """
        np.random.seed(42)
        x_np = np.random.randn(2, 3, 4, 4).astype("float32")
        weight_np = np.random.randn(3, 6, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight)
        loss = out.sum()
        loss.backward()

        # Gradients should be computed
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(weight.grad.shape, weight.shape)
        self.assertEqual(out.shape, (2, 6, 6, 6))  # default padding=0, stride=1

    def test_conv2d_transpose_grad_with_bias(self):
        """测试带偏置的 2D 转置卷积梯度
        Test 2D conv transpose gradient with bias
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 3, 3).astype("float32")
        weight_np = np.random.randn(2, 4, 3, 3).astype("float32")
        bias_np = np.random.randn(4).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)
        bias = paddle.to_tensor(bias_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, bias=bias)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        self.assertEqual(bias.grad.shape, (4,))

    def test_conv2d_transpose_grad_stride(self):
        """测试带步长的 2D 转置卷积梯度
        Test 2D conv transpose gradient with stride
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 3, 4, 4).astype("float32")
        weight_np = np.random.randn(3, 3, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, stride=2)
        loss = out.sum()
        loss.backward()

        # With stride=2, output size = (4-1)*2 + 3 = 9
        self.assertEqual(out.shape, (1, 3, 9, 9))
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_conv2d_transpose_grad_padding(self):
        """测试带填充的 2D 转置卷积梯度
        Test 2D conv transpose gradient with padding
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 5, 5).astype("float32")
        weight_np = np.random.randn(2, 4, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, padding=1)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(out.shape, (1, 4, 5, 5))

    def test_conv2d_transpose_grad_dilation(self):
        """测试带膨胀的 2D 转置卷积梯度
        Test 2D conv transpose gradient with dilation
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 1, 3, 3).astype("float32")
        weight_np = np.random.randn(1, 1, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, dilation=2)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_conv2d_transpose_grad_output_padding(self):
        """测试带输出填充的 2D 转置卷积梯度
        Test 2D conv transpose gradient with output_padding
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 3, 3).astype("float32")
        weight_np = np.random.randn(2, 4, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, stride=2, output_padding=1)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)


class TestConv2dTransposeGradKernelFloat64(unittest.TestCase):
    """float64 类型的 2D 转置卷积梯度测试
    2D Conv Transpose gradient tests with float64"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_conv2d_transpose_grad_float64(self):
        """测试 float64 类型的 2D 转置卷积梯度
        Test 2D conv transpose gradient with float64
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 3, 3).astype("float64")
        weight_np = np.random.randn(2, 2, 3, 3).astype("float64")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight)
        loss = out.sum()
        loss.backward()

        self.assertEqual(out.dtype, paddle.float64)
        self.assertEqual(x.grad.dtype, paddle.float64)
        self.assertEqual(weight.grad.dtype, paddle.float64)


class TestConv2dTransposeGradKernelGroups(unittest.TestCase):
    """分组转置卷积梯度测试 / Grouped conv transpose gradient tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_conv2d_transpose_grad_groups(self):
        """测试分组 2D 转置卷积梯度
        Test grouped 2D conv transpose gradient
        """
        np.random.seed(42)
        # conv_transpose2d weight shape: [in_channels, out_channels/groups, kH, kW]
        x_np = np.random.randn(1, 4, 4, 4).astype("float32")
        weight_np = np.random.randn(4, 4, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, groups=2)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)
        self.assertEqual(out.shape[1], 8)

    def test_depthwise_conv2d_transpose_grad(self):
        """测试深度wise转置卷积梯度
        Test depthwise conv2d transpose gradient
        """
        np.random.seed(42)
        # Depthwise: groups = in_channels = out_channels
        in_channels = 3
        # For depthwise conv_transpose, weight shape is [C, 1, kH, kW]
        x_np = np.random.randn(1, in_channels, 4, 4).astype("float32")
        weight_np = np.random.randn(in_channels, 1, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, groups=in_channels)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)


class TestConv2dTransposeGradKernelPadding(unittest.TestCase):
    """不同填充设置的转置卷积梯度测试
    Conv Transpose gradient tests with different padding settings"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_conv2d_transpose_grad_asymmetric_padding(self):
        """测试非对称填充的梯度
        Test gradient with asymmetric padding
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 4, 4).astype("float32")
        weight_np = np.random.randn(2, 4, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, padding=[1, 2])
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_conv2d_transpose_grad_large_padding(self):
        """测试大填充的梯度
        Test gradient with large padding
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 3, 3).astype("float32")
        weight_np = np.random.randn(2, 4, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, padding=3)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)


class TestConv2dTransposeGradKernelChannelsLast(unittest.TestCase):
    """ChannelsLast 格式的转置卷积梯度测试
    Conv Transpose gradient tests with channels_last format"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_conv2d_transpose_grad_channels_last(self):
        """测试 channels_last 格式的梯度
        Test gradient with channels_last memory format
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 4, 4, 2).astype("float32")
        weight_np = np.random.randn(2, 4, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight, data_format="NHWC")
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)


class TestConv2dTransposeGradKernelEdgeCases(unittest.TestCase):
    """转置卷积梯度边界情况测试
    Conv Transpose gradient edge case tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_conv2d_transpose_grad_1x1_weight(self):
        """测试 1x1 卷积核的转置卷积梯度
        Test conv transpose gradient with 1x1 kernel
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 3, 4, 4).astype("float32")
        weight_np = np.random.randn(3, 6, 1, 1).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight)
        loss = out.sum()
        loss.backward()

        self.assertEqual(out.shape, (1, 6, 4, 4))
        self.assertIsNotNone(x.grad)

    def test_conv2d_transpose_grad_large_kernel(self):
        """测试大卷积核的转置卷积梯度
        Test conv transpose gradient with large kernel
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 1, 8, 8).astype("float32")
        weight_np = np.random.randn(1, 1, 7, 7).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_conv2d_transpose_grad_no_bias(self):
        """测试无偏置的转置卷积梯度
        Test conv transpose gradient without bias (default)
        """
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 3, 3).astype("float32")
        weight_np = np.random.randn(2, 4, 3, 3).astype("float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        weight = paddle.to_tensor(weight_np, stop_gradient=False)

        out = F.conv_transpose2d(x, weight)
        loss = out.sum()
        loss.backward()

        self.assertEqual(out.shape, (1, 4, 5, 5))


if __name__ == "__main__":
    unittest.main()
