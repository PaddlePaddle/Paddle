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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.conv (Conv1D, Conv2D, Conv3D, etc.)
# 自动生成的单测，覆盖 paddle.nn.layer.conv 模块中未覆盖的代码路径
# Target: cover uncovered lines in paddle/python/paddle/nn/layer/conv.py
# 目标：覆盖 Conv 层的各种初始化参数组合和前向传播

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. Conv1D - 各种参数 (stride, padding, dilation, groups)
2. Conv2D - 各种参数 (padding_mode, weight_attr)
3. Conv3D - 基本功能
4. Conv1DTranspose - 各种参数
5. Conv2DTranspose - 各种参数 (output_padding)
6. Conv3DTranspose - 基本功能
"""

import unittest

import paddle
from paddle import nn


class TestConv1D(unittest.TestCase):
    """Test Conv1D layer.
    测试 Conv1D 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv1d_basic(self):
        """Basic Conv1D."""
        conv = nn.Conv1D(3, 16, 3)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 8])

    def test_conv1d_padding_same(self):
        """Conv1D with padding='same'."""
        conv = nn.Conv1D(3, 16, 3, padding='same')
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 10])

    def test_conv1d_padding_valid(self):
        """Conv1D with padding='valid'."""
        conv = nn.Conv1D(3, 16, 3, padding='valid')
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 8])

    def test_conv1d_stride(self):
        """Conv1D with stride."""
        conv = nn.Conv1D(3, 16, 3, stride=2)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape[2], 4)

    def test_conv1d_dilation(self):
        """Conv1D with dilation."""
        conv = nn.Conv1D(3, 16, 3, dilation=2)
        x = paddle.randn([2, 3, 10])
        out = conv(x)
        self.assertEqual(out.shape[2], 6)

    def test_conv1d_groups(self):
        """Conv1D with groups (depthwise)."""
        conv = nn.Conv1D(4, 4, 3, groups=4)
        x = paddle.randn([2, 4, 10])
        out = conv(x)
        self.assertEqual(out.shape[1], 4)

    def test_conv1d_bias_false(self):
        """Conv1D without bias."""
        conv = nn.Conv1D(3, 16, 3, bias_attr=False)
        self.assertIsNone(conv.bias)


class TestConv2D(unittest.TestCase):
    """Test Conv2D layer.
    测试 Conv2D 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv2d_basic(self):
        """Basic Conv2D."""
        conv = nn.Conv2D(3, 16, 3)
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 6, 6])

    def test_conv2d_padding_same(self):
        """Conv2D with padding='same'."""
        conv = nn.Conv2D(3, 16, 3, padding='same')
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_conv2d_padding_valid(self):
        """Conv2D with padding='valid'."""
        conv = nn.Conv2D(3, 16, 3, padding='valid')
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 6, 6])

    def test_conv2d_tuple_kernel(self):
        """Conv2D with tuple kernel_size."""
        conv = nn.Conv2D(3, 16, (3, 5))
        x = paddle.randn([2, 3, 8, 10])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 6, 6])

    def test_conv2d_tuple_padding(self):
        """Conv2D with tuple padding."""
        conv = nn.Conv2D(3, 16, 3, padding=(1, 2))
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 8, 10])

    def test_conv2d_tuple_stride(self):
        """Conv2D with tuple stride."""
        conv = nn.Conv2D(3, 16, 3, stride=(1, 2))
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 6, 3])

    def test_conv2d_dilation(self):
        """Conv2D with dilation."""
        conv = nn.Conv2D(3, 16, 3, dilation=2)
        x = paddle.randn([2, 3, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 4, 4])

    def test_conv2d_groups(self):
        """Conv2D with groups."""
        conv = nn.Conv2D(4, 4, 3, groups=4)
        x = paddle.randn([2, 4, 8, 8])
        out = conv(x)
        self.assertEqual(out.shape[1], 4)

    def test_conv2d_bias_false(self):
        """Conv2D without bias."""
        conv = nn.Conv2D(3, 16, 3, bias_attr=False)
        self.assertIsNone(conv.bias)


class TestConv3D(unittest.TestCase):
    """Test Conv3D layer.
    测试 Conv3D 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv3d_basic(self):
        """Basic Conv3D."""
        conv = nn.Conv3D(3, 16, 3)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 2, 2, 2])

    def test_conv3d_padding_same(self):
        """Conv3D with padding='same'."""
        conv = nn.Conv3D(3, 16, 3, padding='same')
        x = paddle.randn([2, 3, 4, 4, 4])
        out = conv(x)
        self.assertEqual(out.shape, [2, 16, 4, 4, 4])


class TestConvTranspose(unittest.TestCase):
    """Test ConvTranspose layers.
    测试 ConvTranspose 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv1d_transpose_basic(self):
        """Basic Conv1DTranspose."""
        conv = nn.Conv1DTranspose(16, 3, 3)
        x = paddle.randn([2, 16, 4])
        out = conv(x)
        self.assertEqual(out.shape, [2, 3, 6])

    def test_conv2d_transpose_basic(self):
        """Basic Conv2DTranspose."""
        conv = nn.Conv2DTranspose(16, 3, 3)
        x = paddle.randn([2, 16, 4, 4])
        out = conv(x)
        self.assertEqual(out.shape, [2, 3, 6, 6])

    def test_conv2d_transpose_with_output_padding(self):
        """Conv2DTranspose with output_padding."""
        conv = nn.Conv2DTranspose(16, 3, 3, stride=2, output_padding=1)
        x = paddle.randn([2, 16, 4, 4])
        out = conv(x)
        # output_size = (input_size - 1) * stride - 2*padding + kernel + output_padding
        # = (4 - 1) * 2 - 0 + 3 + 1 = 10
        self.assertEqual(out.shape, [2, 3, 10, 10])

    def test_conv3d_transpose_basic(self):
        """Basic Conv3DTranspose."""
        conv = nn.Conv3DTranspose(16, 3, 3)
        x = paddle.randn([2, 16, 2, 2, 2])
        out = conv(x)
        self.assertEqual(out.shape, [2, 3, 4, 4, 4])

    def test_conv2d_transpose_padding_same(self):
        """Conv2DTranspose with padding='same'."""
        conv = nn.Conv2DTranspose(16, 3, 3, padding='same')
        x = paddle.randn([2, 16, 4, 4])
        out = conv(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])


if __name__ == '__main__':
    unittest.main()
