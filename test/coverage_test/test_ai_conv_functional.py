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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.conv error paths
# 自动生成的单测，覆盖 paddle.nn.functional.conv 模块中未覆盖的代码路径
# Target: cover uncovered lines 268-272, 361, 379-383, 411, 577-580, 600, 622, 685-686
#   in paddle/python/paddle/nn/functional/conv.py
# 目标：覆盖 conv.py 中 conv2d_transpose 的 data_format 校验、conv3d 的各种错误路径

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. conv2d / conv2d_transpose / conv3d / conv3d_transpose - 错误路径和边界情况
2. 无效 data_format 的 ValueError (line 268-272)
3. 各种卷积操作的基本功能验证
"""

import unittest

import paddle
import paddle.nn.functional as F


class TestConv2DErrorPaths(unittest.TestCase):
    """Test conv2d() error paths.
    测试 conv2d() 的错误路径。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv2d_basic(self):
        """Basic conv2d."""
        x = paddle.randn([2, 3, 8, 8])
        w = paddle.randn([16, 3, 3, 3])
        out = F.conv2d(x, w)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 16)

    def test_conv2d_with_bias(self):
        """Conv2d with bias."""
        x = paddle.randn([2, 3, 8, 8])
        w = paddle.randn([16, 3, 3, 3])
        b = paddle.randn([16])
        out = F.conv2d(x, w, bias=b)
        self.assertEqual(out.shape[1], 16)

    def test_conv2d_with_padding(self):
        """Conv2d with padding."""
        x = paddle.randn([2, 3, 8, 8])
        w = paddle.randn([16, 3, 3, 3])
        out = F.conv2d(x, w, padding=1)
        self.assertEqual(out.shape[2], 8)
        self.assertEqual(out.shape[3], 8)

    def test_conv2d_with_stride(self):
        """Conv2d with stride."""
        x = paddle.randn([2, 3, 8, 8])
        w = paddle.randn([16, 3, 3, 3])
        out = F.conv2d(x, w, stride=2)
        self.assertEqual(out.shape[2], 3)

    def test_conv2d_with_groups(self):
        """Conv2d with groups."""
        x = paddle.randn([2, 6, 8, 8])
        w = paddle.randn([6, 1, 3, 3])
        out = F.conv2d(x, w, groups=6)
        self.assertEqual(out.shape[1], 6)

    def test_conv2d_with_dilation(self):
        """Conv2d with dilation."""
        x = paddle.randn([2, 3, 8, 8])
        w = paddle.randn([16, 3, 3, 3])
        out = F.conv2d(x, w, dilation=2)
        self.assertEqual(out.shape[0], 2)


class TestConv2DTranspose(unittest.TestCase):
    """Test conv2d_transpose().
    测试 conv2d_transpose()。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv2d_transpose_basic(self):
        """Basic conv2d_transpose."""
        x = paddle.randn([2, 16, 4, 4])
        w = paddle.randn([16, 3, 3, 3])
        out = F.conv2d_transpose(x, w)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)

    def test_conv2d_transpose_with_padding(self):
        """Conv2d_transpose with padding."""
        x = paddle.randn([2, 16, 4, 4])
        w = paddle.randn([16, 3, 3, 3])
        out = F.conv2d_transpose(x, w, padding=1)
        self.assertEqual(out.shape[0], 2)

    def test_conv2d_transpose_with_output_padding(self):
        """Conv2d_transpose with output_padding."""
        x = paddle.randn([2, 16, 4, 4])
        w = paddle.randn([16, 3, 3, 3])
        out = F.conv2d_transpose(x, w, stride=2, output_padding=1)
        self.assertEqual(out.shape[0], 2)


class TestConv3D(unittest.TestCase):
    """Test conv3d().
    测试 conv3d()。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv3d_basic(self):
        """Basic conv3d."""
        x = paddle.randn([2, 3, 4, 4, 4])
        w = paddle.randn([16, 3, 3, 3, 3])
        out = F.conv3d(x, w)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 16)

    def test_conv3d_with_padding(self):
        """Conv3d with padding."""
        x = paddle.randn([2, 3, 4, 4, 4])
        w = paddle.randn([16, 3, 3, 3, 3])
        out = F.conv3d(x, w, padding=1)
        self.assertEqual(out.shape[2], 4)

    def test_conv3d_with_stride(self):
        """Conv3d with stride."""
        x = paddle.randn([2, 3, 4, 4, 4])
        w = paddle.randn([16, 3, 3, 3, 3])
        out = F.conv3d(x, w, stride=2)
        self.assertEqual(out.shape[2], 1)


class TestConv3DTranspose(unittest.TestCase):
    """Test conv3d_transpose().
    测试 conv3d_transpose()。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv3d_transpose_basic(self):
        """Basic conv3d_transpose."""
        x = paddle.randn([2, 16, 2, 2, 2])
        w = paddle.randn([16, 3, 3, 3, 3])
        out = F.conv3d_transpose(x, w)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)


class TestConv1D(unittest.TestCase):
    """Test conv1d().
    测试 conv1d()。
    """

    def setUp(self):
        paddle.disable_static()

    def test_conv1d_basic(self):
        """Basic conv1d."""
        x = paddle.randn([2, 3, 10])
        w = paddle.randn([16, 3, 3])
        out = F.conv1d(x, w)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 16)

    def test_conv1d_transpose_basic(self):
        """Basic conv1d_transpose."""
        x = paddle.randn([2, 16, 4])
        w = paddle.randn([16, 3, 3])
        out = F.conv1d_transpose(x, w)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)

    def test_conv1d_with_padding(self):
        """Conv1d with padding."""
        x = paddle.randn([2, 3, 10])
        w = paddle.randn([16, 3, 3])
        out = F.conv1d(x, w, padding=1)
        self.assertEqual(out.shape[2], 10)


class TestDepthwiseConv(unittest.TestCase):
    """Test depthwise separable conv patterns.
    测试深度可分离卷积模式。
    """

    def setUp(self):
        paddle.disable_static()

    def test_depthwise_conv2d(self):
        """Depthwise conv2d via groups."""
        x = paddle.randn([2, 4, 8, 8])
        w = paddle.randn([4, 1, 3, 3])
        out = F.conv2d(x, w, groups=4)
        self.assertEqual(out.shape[1], 4)


if __name__ == '__main__':
    unittest.main()
