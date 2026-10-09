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
池化层单元测试 / Pooling Layers Unit Tests

测试目标 / Test Target:
  paddle.nn 池化层 (AvgPool, MaxPool, AdaptivePool等)

覆盖的模块 / Covered Modules:
  - paddle.nn.AvgPool1D/2D/3D: 平均池化
  - paddle.nn.MaxPool1D/2D/3D: 最大池化
  - paddle.nn.AdaptiveAvgPool1D/2D/3D: 自适应平均池化
  - paddle.nn.AdaptiveMaxPool1D/2D/3D: 自适应最大池化

作用 / Purpose:
  补充池化层API的各种参数组合测试，提升覆盖率。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


class TestAvgPoolLayers(unittest.TestCase):
    """测试平均池化层 / Test average pooling layers"""

    def test_avgpool1d(self):
        """测试AvgPool1D / Test AvgPool1D"""
        pool = nn.AvgPool1D(kernel_size=2, stride=2)
        x = paddle.randn([4, 3, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 8])

    def test_avgpool1d_padding(self):
        """测试带填充AvgPool1D / Test AvgPool1D with padding"""
        pool = nn.AvgPool1D(kernel_size=3, stride=1, padding=1)
        x = paddle.randn([4, 3, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 16])

    def test_avgpool2d(self):
        """测试AvgPool2D / Test AvgPool2D"""
        pool = nn.AvgPool2D(kernel_size=2, stride=2)
        x = paddle.randn([4, 3, 16, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 8, 8])

    def test_avgpool2d_ceil_mode(self):
        """测试ceil模式AvgPool2D / Test AvgPool2D with ceil mode"""
        pool = nn.AvgPool2D(kernel_size=3, stride=2, ceil_mode=True)
        x = paddle.randn([4, 3, 15, 15])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 7, 7])

    def test_avgpool3d(self):
        """测试AvgPool3D / Test AvgPool3D"""
        pool = nn.AvgPool3D(kernel_size=2, stride=2)
        x = paddle.randn([2, 3, 8, 8, 8])
        y = pool(x)
        self.assertEqual(y.shape, [2, 3, 4, 4, 4])


class TestMaxPoolLayers(unittest.TestCase):
    """测试最大池化层 / Test max pooling layers"""

    def test_maxpool1d(self):
        """测试MaxPool1D / Test MaxPool1D"""
        pool = nn.MaxPool1D(kernel_size=2, stride=2)
        x = paddle.randn([4, 3, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 8])

    def test_maxpool1d_return_mask(self):
        """测试返回掩码的MaxPool1D / Test MaxPool1D with return mask"""
        pool = nn.MaxPool1D(kernel_size=2, stride=2, return_mask=True)
        x = paddle.randn([4, 3, 16])
        y, idx = pool(x)
        self.assertEqual(y.shape, [4, 3, 8])
        self.assertEqual(idx.shape, [4, 3, 8])

    def test_maxpool2d(self):
        """测试MaxPool2D / Test MaxPool2D"""
        pool = nn.MaxPool2D(kernel_size=2, stride=2)
        x = paddle.randn([4, 3, 16, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 8, 8])

    def test_maxpool2d_ceil_mode(self):
        """测试ceil模式MaxPool2D / Test MaxPool2D with ceil mode"""
        pool = nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True)
        x = paddle.randn([4, 3, 15, 15])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 7, 7])

    def test_maxpool3d(self):
        """测试MaxPool3D / Test MaxPool3D"""
        pool = nn.MaxPool3D(kernel_size=2, stride=2)
        x = paddle.randn([2, 3, 8, 8, 8])
        y = pool(x)
        self.assertEqual(y.shape, [2, 3, 4, 4, 4])


class TestAdaptivePoolLayers(unittest.TestCase):
    """测试自适应池化层 / Test adaptive pooling layers"""

    def test_adaptive_avgpool1d(self):
        """测试AdaptiveAvgPool1D / Test AdaptiveAvgPool1D"""
        pool = nn.AdaptiveAvgPool1D(output_size=8)
        x = paddle.randn([4, 3, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 8])

    def test_adaptive_avgpool2d(self):
        """测试AdaptiveAvgPool2D / Test AdaptiveAvgPool2D"""
        pool = nn.AdaptiveAvgPool2D(output_size=(8, 8))
        x = paddle.randn([4, 3, 16, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 8, 8])

    def test_adaptive_avgpool2d_global(self):
        """测试全局自适应平均池化 / Test global adaptive avg pool"""
        pool = nn.AdaptiveAvgPool2D(output_size=1)
        x = paddle.randn([4, 3, 16, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 1, 1])

    def test_adaptive_avgpool3d(self):
        """测试AdaptiveAvgPool3D / Test AdaptiveAvgPool3D"""
        pool = nn.AdaptiveAvgPool3D(output_size=(4, 4, 4))
        x = paddle.randn([2, 3, 8, 8, 8])
        y = pool(x)
        self.assertEqual(y.shape, [2, 3, 4, 4, 4])

    def test_adaptive_maxpool2d(self):
        """测试AdaptiveMaxPool2D / Test AdaptiveMaxPool2D"""
        pool = nn.AdaptiveMaxPool2D(output_size=(8, 8))
        x = paddle.randn([4, 3, 16, 16])
        y = pool(x)
        self.assertEqual(y.shape, [4, 3, 8, 8])


if __name__ == '__main__':
    unittest.main()
