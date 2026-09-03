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
卷积层单元测试 / Convolution Layers Unit Tests

测试目标 / Test Target:
  paddle.nn 卷积层 (paddle.nn.Conv1D, Conv2D, Conv3D, ConvTranspose)

覆盖的模块 / Covered Modules:
  - paddle.nn.Conv1D: 1D卷积层
  - paddle.nn.Conv2D: 2D卷积层
  - paddle.nn.Conv3D: 3D卷积层
  - paddle.nn.Conv1DTranspose: 1D转置卷积
  - paddle.nn.Conv2DTranspose: 2D转置卷积
  - paddle.nn.Conv3DTranspose: 3D转置卷积

作用 / Purpose:
  补充卷积层API的各种参数组合测试，提升覆盖率。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


class TestConv1DLayer(unittest.TestCase):
    """测试Conv1D层 / Test Conv1D layer"""

    def test_conv1d_basic(self):
        """测试基本Conv1D / Test basic Conv1D"""
        conv = nn.Conv1D(3, 8, 3)
        x = paddle.randn([4, 3, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 14])

    def test_conv1d_padding(self):
        """测试带填充Conv1D / Test Conv1D with padding"""
        conv = nn.Conv1D(3, 8, 3, padding=1)
        x = paddle.randn([4, 3, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 16])

    def test_conv1d_stride(self):
        """测试带步幅Conv1D / Test Conv1D with stride"""
        conv = nn.Conv1D(3, 8, 3, stride=2)
        x = paddle.randn([4, 3, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 7])

    def test_conv1d_dilation(self):
        """测试空洞Conv1D / Test dilated Conv1D"""
        conv = nn.Conv1D(3, 8, 3, dilation=2)
        x = paddle.randn([4, 3, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 12])

    def test_conv1d_groups(self):
        """测试分组Conv1D / Test grouped Conv1D"""
        conv = nn.Conv1D(6, 6, 3, groups=3)
        x = paddle.randn([4, 6, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 6, 14])

    def test_conv1d_no_bias(self):
        """测试无偏置Conv1D / Test Conv1D without bias"""
        conv = nn.Conv1D(3, 8, 3, bias_attr=False)
        x = paddle.randn([4, 3, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 14])
        self.assertIsNone(conv.bias)


class TestConv2DLayer(unittest.TestCase):
    """测试Conv2D层 / Test Conv2D layer"""

    def test_conv2d_basic(self):
        """测试基本Conv2D / Test basic Conv2D"""
        conv = nn.Conv2D(3, 16, 3)
        x = paddle.randn([4, 3, 32, 32])
        y = conv(x)
        self.assertEqual(y.shape, [4, 16, 30, 30])

    def test_conv2d_depthwise(self):
        """测试深度卷积 / Test depthwise convolution"""
        conv = nn.Conv2D(8, 8, 3, groups=8)
        x = paddle.randn([4, 8, 16, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 14, 14])

    def test_conv2d_padding_same(self):
        """测试same填充Conv2D / Test Conv2D with SAME padding"""
        conv = nn.Conv2D(3, 8, 3, padding='SAME')
        x = paddle.randn([4, 3, 16, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 16, 16])

    def test_conv2d_dilation(self):
        """测试空洞Conv2D / Test dilated Conv2D"""
        conv = nn.Conv2D(3, 8, 3, dilation=2)
        x = paddle.randn([4, 3, 16, 16])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 12, 12])


class TestConv3DLayer(unittest.TestCase):
    """测试Conv3D层 / Test Conv3D layer"""

    def test_conv3d_basic(self):
        """测试基本Conv3D / Test basic Conv3D"""
        conv = nn.Conv3D(3, 8, 3)
        x = paddle.randn([2, 3, 8, 16, 16])
        y = conv(x)
        self.assertEqual(y.shape, [2, 8, 6, 14, 14])

    def test_conv3d_padding(self):
        """测试带填充Conv3D / Test Conv3D with padding"""
        conv = nn.Conv3D(3, 8, 3, padding=1)
        x = paddle.randn([2, 3, 8, 8, 8])
        y = conv(x)
        self.assertEqual(y.shape, [2, 8, 8, 8, 8])


class TestConvTransposeLayers(unittest.TestCase):
    """测试转置卷积层 / Test transposed convolution layers"""

    def test_conv1d_transpose(self):
        """测试Conv1DTranspose / Test Conv1DTranspose"""
        conv = nn.Conv1DTranspose(8, 3, 3)
        x = paddle.randn([4, 8, 14])
        y = conv(x)
        self.assertEqual(y.shape, [4, 3, 16])

    def test_conv2d_transpose(self):
        """测试Conv2DTranspose / Test Conv2DTranspose"""
        conv = nn.Conv2DTranspose(16, 8, 3)
        x = paddle.randn([4, 16, 7, 7])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 9, 9])

    def test_conv2d_transpose_stride(self):
        """测试带步幅Conv2DTranspose / Test Conv2DTranspose with stride"""
        conv = nn.Conv2DTranspose(16, 8, 4, stride=2)
        x = paddle.randn([4, 16, 7, 7])
        y = conv(x)
        self.assertEqual(y.shape, [4, 8, 16, 16])

    def test_conv3d_transpose(self):
        """测试Conv3DTranspose / Test Conv3DTranspose"""
        conv = nn.Conv3DTranspose(8, 4, 3)
        x = paddle.randn([2, 8, 6, 6, 6])
        y = conv(x)
        self.assertEqual(y.shape, [2, 4, 8, 8, 8])


if __name__ == '__main__':
    unittest.main()
