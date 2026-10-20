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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.pooling (Pool2D, Pool3D, Adaptive*)
# 自动生成的单测，覆盖 paddle.nn.layer.pooling 模块中未覆盖的代码路径
# Target: cover uncovered lines in paddle/python/paddle/nn/layer/pooling.py
# 目标：覆盖 Pool 层的各种初始化参数和前向传播

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. MaxPool2D - padding_mode, return_mask, ceil_mode
2. AvgPool2D - exclusive, ceil_mode, count_include_pad
3. MaxPool3D / AvgPool3D - 基本功能
4. AdaptiveAvgPool1D / AdaptiveAvgPool2D / AdaptiveAvgPool3D
5. AdaptiveMaxPool1D / AdaptiveMaxPool2D
6. MaxPool1D / AvgPool1D
"""

import unittest

import paddle
from paddle import nn


class TestMaxPool2D(unittest.TestCase):
    """Test MaxPool2D layer.
    测试 MaxPool2D 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_maxpool2d_basic(self):
        """Basic MaxPool2D."""
        pool = nn.MaxPool2D(kernel_size=2, stride=2)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_maxpool2d_padding(self):
        """MaxPool2D with padding."""
        pool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_maxpool2d_same(self):
        """MaxPool2D with padding='same'."""
        pool = nn.MaxPool2D(kernel_size=2, stride=2, padding='same')
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_maxpool2d_valid(self):
        """MaxPool2D with padding='valid'."""
        pool = nn.MaxPool2D(kernel_size=2, stride=2, padding='valid')
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_maxpool2d_ceil_mode(self):
        """MaxPool2D with ceil_mode=True."""
        pool = nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertTrue(out.shape[2] >= 3)

    def test_maxpool2d_return_mask(self):
        """MaxPool2D with return_mask=True."""
        pool = nn.MaxPool2D(kernel_size=2, stride=2, return_mask=True)
        x = paddle.randn([2, 3, 8, 8])
        out, mask = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])
        self.assertEqual(mask.shape, [2, 3, 4, 4])

    def test_maxpool2d_global(self):
        """MaxPool2D as global pooling."""
        pool = nn.MaxPool2D(kernel_size=8)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 1, 1])


class TestAvgPool2D(unittest.TestCase):
    """Test AvgPool2D layer.
    测试 AvgPool2D 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_avgpool2d_basic(self):
        """Basic AvgPool2D."""
        pool = nn.AvgPool2D(kernel_size=2, stride=2)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_avgpool2d_padding(self):
        """AvgPool2D with padding."""
        pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_avgpool2d_same(self):
        """AvgPool2D with padding='same'."""
        pool = nn.AvgPool2D(kernel_size=2, stride=2, padding='same')
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_avgpool2d_global(self):
        """AvgPool2D as global pooling."""
        pool = nn.AvgPool2D(kernel_size=8)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 1, 1])

    def test_avgpool2d_exclusive(self):
        """AvgPool2D with exclusive=True (default)."""
        pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=True)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_avgpool2d_count_include_pad(self):
        """AvgPool2D with count_include_pad=True."""
        pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 8, 8])


class TestPool3D(unittest.TestCase):
    """Test 3D pooling layers.
    测试 3D 池化层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_maxpool3d_basic(self):
        """Basic MaxPool3D."""
        pool = nn.MaxPool3D(kernel_size=2, stride=2)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])

    def test_avgpool3d_basic(self):
        """Basic AvgPool3D."""
        pool = nn.AvgPool3D(kernel_size=2, stride=2)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])


class TestPool1D(unittest.TestCase):
    """Test 1D pooling layers.
    测试 1D 池化层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_maxpool1d_basic(self):
        """Basic MaxPool1D."""
        pool = nn.MaxPool1D(kernel_size=2, stride=2)
        x = paddle.randn([2, 3, 10])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 5])

    def test_avgpool1d_basic(self):
        """Basic AvgPool1D."""
        pool = nn.AvgPool1D(kernel_size=2, stride=2)
        x = paddle.randn([2, 3, 10])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 5])


class TestAdaptivePool(unittest.TestCase):
    """Test Adaptive pooling layers.
    测试自适应池化层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_adaptive_avg_pool1d(self):
        """AdaptiveAvgPool1D."""
        pool = nn.AdaptiveAvgPool1D(output_size=4)
        x = paddle.randn([2, 3, 10])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_adaptive_avg_pool2d(self):
        """AdaptiveAvgPool2D."""
        pool = nn.AdaptiveAvgPool2D(output_size=4)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_adaptive_avg_pool3d(self):
        """AdaptiveAvgPool3D."""
        pool = nn.AdaptiveAvgPool3D(output_size=2)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])

    def test_adaptive_max_pool1d(self):
        """AdaptiveMaxPool1D."""
        pool = nn.AdaptiveMaxPool1D(output_size=4)
        x = paddle.randn([2, 3, 10])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_adaptive_max_pool2d(self):
        """AdaptiveMaxPool2D."""
        pool = nn.AdaptiveMaxPool2D(output_size=4)
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_adaptive_avg_pool2d_tuple(self):
        """AdaptiveAvgPool2D with tuple output_size."""
        pool = nn.AdaptiveAvgPool2D(output_size=(4, 6))
        x = paddle.randn([2, 3, 8, 8])
        out = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 6])

    def test_adaptive_max_pool2d_with_return_mask(self):
        """AdaptiveMaxPool2D with return_mask."""
        pool = nn.AdaptiveMaxPool2D(output_size=4, return_mask=True)
        x = paddle.randn([2, 3, 8, 8])
        out, mask = pool(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])


if __name__ == '__main__':
    unittest.main()
