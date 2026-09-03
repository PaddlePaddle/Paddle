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
函数式卷积操作单元测试 / Functional Convolution Unit Tests

测试目标 / Test Target:
  paddle.nn.functional.conv 模块 (覆盖率约83.4%)

覆盖的模块 / Covered Modules:
  - paddle.nn.functional.conv1d: 1D卷积
  - paddle.nn.functional.conv2d: 2D卷积
  - paddle.nn.functional.conv3d: 3D卷积
  - paddle.nn.functional.conv_transpose1d: 1D转置卷积
  - paddle.nn.functional.conv_transpose2d: 2D转置卷积
  - paddle.nn.functional.conv_transpose3d: 3D转置卷积

作用 / Purpose:
  覆盖函数式卷积API的各种参数组合，补充卷积操作的测试覆盖率。
"""

import unittest

import paddle
import paddle.nn.functional as F

paddle.disable_static()


class TestConv1D(unittest.TestCase):
    """测试1D卷积函数 / Test 1D convolution function"""

    def test_conv1d_basic(self):
        """测试基本1D卷积 / Test basic 1D convolution"""
        x = paddle.randn([4, 3, 16])
        weight = paddle.randn([8, 3, 3])
        y = F.conv1d(x, weight)
        self.assertEqual(y.shape, [4, 8, 14])

    def test_conv1d_padding(self):
        """测试带填充的1D卷积 / Test 1D convolution with padding"""
        x = paddle.randn([4, 3, 16])
        weight = paddle.randn([8, 3, 3])
        y = F.conv1d(x, weight, padding=1)
        self.assertEqual(y.shape, [4, 8, 16])

    def test_conv1d_stride(self):
        """测试带步幅的1D卷积 / Test 1D convolution with stride"""
        x = paddle.randn([4, 3, 16])
        weight = paddle.randn([8, 3, 3])
        y = F.conv1d(x, weight, stride=2)
        self.assertEqual(y.shape, [4, 8, 7])

    def test_conv1d_dilation(self):
        """测试空洞1D卷积 / Test dilated 1D convolution"""
        x = paddle.randn([4, 3, 16])
        weight = paddle.randn([8, 3, 3])
        y = F.conv1d(x, weight, dilation=2)
        self.assertEqual(y.shape, [4, 8, 12])

    def test_conv1d_with_bias(self):
        """测试带偏置的1D卷积 / Test 1D convolution with bias"""
        x = paddle.randn([4, 3, 16])
        weight = paddle.randn([8, 3, 3])
        bias = paddle.randn([8])
        y = F.conv1d(x, weight, bias=bias)
        self.assertEqual(y.shape, [4, 8, 14])

    def test_conv1d_groups(self):
        """测试分组1D卷积 / Test grouped 1D convolution"""
        x = paddle.randn([4, 6, 16])
        weight = paddle.randn([6, 2, 3])  # out_channels=6, in_channels/groups=2
        y = F.conv1d(x, weight, groups=3)
        self.assertEqual(y.shape, [4, 6, 14])


class TestConv2D(unittest.TestCase):
    """测试2D卷积函数 / Test 2D convolution function"""

    def test_conv2d_basic(self):
        """测试基本2D卷积 / Test basic 2D convolution"""
        x = paddle.randn([4, 3, 32, 32])
        weight = paddle.randn([16, 3, 3, 3])
        y = F.conv2d(x, weight)
        self.assertEqual(y.shape, [4, 16, 30, 30])

    def test_conv2d_padding_same(self):
        """测试same填充2D卷积 / Test 2D convolution with same padding"""
        x = paddle.randn([4, 3, 16, 16])
        weight = paddle.randn([8, 3, 3, 3])
        y = F.conv2d(x, weight, padding='SAME')
        self.assertEqual(y.shape, [4, 8, 16, 16])

    def test_conv2d_stride(self):
        """测试带步幅2D卷积 / Test 2D convolution with stride"""
        x = paddle.randn([4, 3, 16, 16])
        weight = paddle.randn([8, 3, 3, 3])
        y = F.conv2d(x, weight, stride=2)
        self.assertEqual(y.shape, [4, 8, 7, 7])

    def test_conv2d_dilation(self):
        """测试空洞2D卷积 / Test dilated 2D convolution"""
        x = paddle.randn([4, 3, 16, 16])
        weight = paddle.randn([8, 3, 3, 3])
        y = F.conv2d(x, weight, dilation=2)
        self.assertEqual(y.shape, [4, 8, 12, 12])

    def test_conv2d_depthwise(self):
        """测试深度可分离卷积 / Test depthwise convolution"""
        x = paddle.randn([4, 8, 16, 16])
        weight = paddle.randn([8, 1, 3, 3])  # groups=8 (depthwise)
        y = F.conv2d(x, weight, groups=8)
        self.assertEqual(y.shape, [4, 8, 14, 14])

    def test_conv2d_asymmetric_padding(self):
        """测试非对称填充2D卷积 / Test asymmetric padding 2D convolution"""
        x = paddle.randn([4, 3, 16, 16])
        weight = paddle.randn([8, 3, 3, 3])
        y = F.conv2d(x, weight, padding=1)  # symmetric
        self.assertEqual(y.shape, [4, 8, 16, 16])


class TestConv3D(unittest.TestCase):
    """测试3D卷积函数 / Test 3D convolution function"""

    def test_conv3d_basic(self):
        """测试基本3D卷积 / Test basic 3D convolution"""
        x = paddle.randn([2, 3, 8, 16, 16])
        weight = paddle.randn([8, 3, 3, 3, 3])
        y = F.conv3d(x, weight)
        self.assertEqual(y.shape, [2, 8, 6, 14, 14])

    def test_conv3d_with_padding(self):
        """测试带填充3D卷积 / Test 3D convolution with padding"""
        x = paddle.randn([2, 3, 8, 8, 8])
        weight = paddle.randn([8, 3, 3, 3, 3])
        y = F.conv3d(x, weight, padding=1)
        self.assertEqual(y.shape, [2, 8, 8, 8, 8])


class TestConvTranspose(unittest.TestCase):
    """测试转置卷积 / Test transposed convolution"""

    def test_conv1d_transpose(self):
        """测试1D转置卷积 / Test 1D transposed convolution"""
        x = paddle.randn([4, 8, 14])
        weight = paddle.randn([8, 3, 3])
        y = F.conv1d_transpose(x, weight)
        self.assertEqual(y.shape, [4, 3, 16])

    def test_conv2d_transpose_basic(self):
        """测试基本2D转置卷积 / Test basic 2D transposed convolution"""
        x = paddle.randn([4, 16, 7, 7])
        weight = paddle.randn([16, 8, 3, 3])
        y = F.conv2d_transpose(x, weight)
        self.assertEqual(y.shape, [4, 8, 9, 9])

    def test_conv2d_transpose_stride(self):
        """测试带步幅2D转置卷积 / Test 2D transposed convolution with stride"""
        x = paddle.randn([4, 16, 7, 7])
        weight = paddle.randn([16, 8, 4, 4])
        y = F.conv2d_transpose(x, weight, stride=2)
        self.assertEqual(y.shape, [4, 8, 16, 16])

    def test_conv3d_transpose(self):
        """测试3D转置卷积 / Test 3D transposed convolution"""
        x = paddle.randn([2, 8, 6, 6, 6])
        weight = paddle.randn([8, 4, 3, 3, 3])
        y = F.conv3d_transpose(x, weight)
        self.assertEqual(y.shape, [2, 4, 8, 8, 8])


if __name__ == '__main__':
    unittest.main()
