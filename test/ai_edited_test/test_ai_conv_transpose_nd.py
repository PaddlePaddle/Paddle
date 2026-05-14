# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.conv
# 自动生成的单测，覆盖 paddle.nn.layer.conv 模块中未覆盖的代码
# Target: paddle/nn/layer/conv.py

"""
测试模块：paddle.nn.layer.conv
Test Module: paddle.nn.layer.conv

本测试覆盖以下功能：
This test covers the following functions:
1. Conv1D - 一维卷积 / 1D convolution with different padding modes, NLC format
2. Conv2D - 二维卷积 / 2D convolution with padding modes, NHWC, groups
3. Conv3D - 三维卷积 / 3D convolution with padding modes, NDHWC
4. Conv1DTranspose - 一维转置卷积 / 1D transpose convolution
5. Conv2DTranspose - 二维转置卷积 / 2D transpose convolution with output_size
6. Conv3DTranspose - 三维转置卷积 / 3D transpose convolution
7. _ConvNd extra_repr / Extra repr for conv layers
"""

import unittest

import paddle
from paddle import nn


class TestConv1DComprehensive(unittest.TestCase):
    """测试Conv1D一维卷积
    Test Conv1D"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_conv1d_basic(self):
        """测试基本Conv1D / Test basic Conv1D"""
        conv = nn.Conv1D(3, 6, kernel_size=3)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 8])

    def test_conv1d_nlc(self):
        """测试NLC格式 / Test Conv1D with NLC data_format"""
        conv = nn.Conv1D(3, 6, kernel_size=3, data_format='NLC')
        x = paddle.randn([2, 10, 3])
        out = conv(x)
        self.assertEqual(out.shape, [2, 8, 6])

    def test_conv1d_padding(self):
        """测试padding / Test Conv1D with padding"""
        conv = nn.Conv1D(3, 6, kernel_size=3, padding=1)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 10])

    def test_conv1d_stride(self):
        """测试stride / Test Conv1D with stride"""
        conv = nn.Conv1D(3, 6, kernel_size=3, stride=2)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 4])

    def test_conv1d_dilation(self):
        """测试dilation / Test Conv1D with dilation"""
        conv = nn.Conv1D(3, 6, kernel_size=3, dilation=2)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 6])

    def test_conv1d_groups(self):
        """测试分组卷积 / Test Conv1D with groups"""
        conv = nn.Conv1D(4, 4, kernel_size=3, groups=2)
        x = paddle.randn([2, 4, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_conv1d_no_bias(self):
        """测试无偏置 / Test Conv1D without bias"""
        conv = nn.Conv1D(3, 6, kernel_size=3, bias=False)
        self.assertIsNone(conv.bias)

    def test_conv1d_same_padding(self):
        """测试same padding / Test Conv1D with same padding"""
        conv = nn.Conv1D(3, 6, kernel_size=3, padding='same')
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 10])

    def test_conv1d_valid_padding(self):
        """测试valid padding / Test Conv1D with valid padding"""
        conv = nn.Conv1D(3, 6, kernel_size=3, padding='valid')
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 8])

    def test_conv1d_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        conv = nn.Conv1D(
            3, 6, kernel_size=3, stride=2, padding=1, dilation=1, groups=1
        )
        r = conv.extra_repr()
        self.assertIn('stride', r)
        self.assertIn('padding', r)


class TestConv2DComprehensive(unittest.TestCase):
    """测试Conv2D二维卷积
    Test Conv2D"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_conv2d_basic(self):
        """测试基本Conv2D / Test basic Conv2D"""
        conv = nn.Conv2D(3, 6, kernel_size=3)
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 6, 6])

    def test_conv2d_nhwc(self):
        """测试NHWC格式 / Test Conv2D with NHWC data_format"""
        conv = nn.Conv2D(3, 6, kernel_size=3, data_format='NHWC')
        x = paddle.randn([2, 8, 8, 3])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 6, 6])

    def test_conv2d_groups(self):
        """测试分组卷积 / Test Conv2D with groups"""
        conv = nn.Conv2D(4, 8, kernel_size=3, groups=2)
        x = paddle.randn([2, 4, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 8, 6, 6])

    def test_conv2d_dilation(self):
        """测试dilation / Test Conv2D with dilation"""
        conv = nn.Conv2D(3, 6, kernel_size=3, dilation=2)
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 4, 4])

    def test_conv2d_same_padding(self):
        """测试same padding / Test Conv2D with same padding"""
        conv = nn.Conv2D(3, 6, kernel_size=3, padding='same')
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 8, 8])

    def test_conv2d_no_bias(self):
        """测试无偏置 / Test Conv2D without bias"""
        conv = nn.Conv2D(3, 6, kernel_size=3, bias=False)
        self.assertIsNone(conv.bias)

    def test_conv2d_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        conv = nn.Conv2D(
            3, 6, kernel_size=3, stride=2, padding=1, data_format='NHWC'
        )
        r = conv.extra_repr()
        self.assertIn('NHWC', r)
        self.assertIn('stride', r)


class TestConv3DComprehensive(unittest.TestCase):
    """测试Conv3D三维卷积
    Test Conv3D"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_conv3d_basic(self):
        """测试基本Conv3D / Test basic Conv3D"""
        conv = nn.Conv3D(3, 6, kernel_size=3)
        x = paddle.randn([2, 3, 8, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 6, 6, 6])

    def test_conv3d_ndhwc(self):
        """测试NDHWC格式 / Test Conv3D with NDHWC data_format"""
        conv = nn.Conv3D(3, 6, kernel_size=3, data_format='NDHWC')
        x = paddle.randn([2, 8, 8, 8, 3])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 6, 6, 6])

    def test_conv3d_stride(self):
        """测试stride / Test Conv3D with stride"""
        conv = nn.Conv3D(3, 6, kernel_size=3, stride=2)
        x = paddle.randn([2, 3, 8, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 3, 3, 3])

    def test_conv3d_padding(self):
        """测试padding / Test Conv3D with padding"""
        conv = nn.Conv3D(3, 6, kernel_size=3, padding=1)
        x = paddle.randn([2, 3, 8, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 8, 8, 8])

    def test_conv3d_same_padding(self):
        """测试same padding / Test Conv3D with same padding"""
        conv = nn.Conv3D(3, 6, kernel_size=3, padding='same')
        x = paddle.randn([2, 3, 8, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 8, 8, 8])

    def test_conv3d_no_bias(self):
        """测试无偏置 / Test Conv3D without bias"""
        conv = nn.Conv3D(3, 6, kernel_size=3, bias=False)
        self.assertIsNone(conv.bias)


class TestConvTransposeComprehensive(unittest.TestCase):
    """测试ConvTranspose系列
    Test ConvTranspose family"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_conv1d_transpose_basic(self):
        """测试Conv1DTranspose / Test Conv1DTranspose"""
        conv = nn.Conv1DTranspose(3, 6, kernel_size=3)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 12])

    def test_conv1d_transpose_no_bias(self):
        """测试无偏置的Conv1DTranspose / Test Conv1DTranspose without bias"""
        conv = nn.Conv1DTranspose(3, 6, kernel_size=3, bias_attr=False)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 12])

    def test_conv2d_transpose_basic(self):
        """测试Conv2DTranspose / Test Conv2DTranspose"""
        conv = nn.Conv2DTranspose(3, 6, kernel_size=3)
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 10, 10])

    def test_conv2d_transpose_output_size(self):
        """测试带output_size的Conv2DTranspose / Test Conv2DTranspose with output_size"""
        conv = nn.Conv2DTranspose(3, 6, kernel_size=3, stride=2)
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x, output_size=[18, 18])
        self.assertEqual(out.shape, [2, 6, 18, 18])

    def test_conv2d_transpose_no_bias(self):
        """测试无偏置的Conv2DTranspose / Test Conv2DTranspose without bias"""
        conv = nn.Conv2DTranspose(3, 6, kernel_size=3, bias_attr=False)
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 10, 10])

    def test_conv3d_transpose_basic(self):
        """测试Conv3DTranspose / Test Conv3DTranspose"""
        conv = nn.Conv3DTranspose(3, 6, kernel_size=3)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 6, 6, 6])

    def test_conv3d_transpose_output_size(self):
        """测试带output_size的Conv3DTranspose / Test Conv3DTranspose with output_size"""
        conv = nn.Conv3DTranspose(3, 6, kernel_size=3, stride=2)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = conv(x, output_size=[9, 9, 9])
        self.assertEqual(out.shape, [2, 6, 9, 9, 9])

    def test_conv3d_transpose_no_bias(self):
        """测试无偏置的Conv3DTranspose / Test Conv3DTranspose without bias"""
        conv = nn.Conv3DTranspose(3, 6, kernel_size=3, bias_attr=False)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = conv(x)
        self.assertEqual(out.shape, [2, 6, 6, 6, 6])


class TestConvErrorHandling(unittest.TestCase):
    """测试卷积层错误处理
    Test conv layer error handling"""

    def test_invalid_padding_mode(self):
        """测试无效padding_mode / Test invalid padding_mode"""
        with self.assertRaises(ValueError):
            nn.Conv2D(3, 6, 3, padding_mode='invalid')

    def test_invalid_data_format(self):
        """测试无效data_format / Test invalid data_format"""
        with self.assertRaises(ValueError):
            nn.Conv2D(3, 6, 3, data_format='INVALID')

    def test_weight_attr_false(self):
        """测试weight_attr=False / Test weight_attr=False"""
        with self.assertRaises(AssertionError):
            nn.Conv2D(3, 6, 3, weight_attr=False)


if __name__ == '__main__':
    unittest.main()
