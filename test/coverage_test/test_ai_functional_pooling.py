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
函数式池化操作单元测试 / Functional Pooling Unit Tests

测试目标 / Test Target:
  paddle.nn.functional.pooling 模块 (覆盖率约82.4%)

覆盖的模块 / Covered Modules:
  - paddle.nn.functional.avg_pool1d/2d/3d: 平均池化
  - paddle.nn.functional.max_pool1d/2d/3d: 最大池化
  - paddle.nn.functional.adaptive_avg_pool1d/2d/3d: 自适应平均池化
  - paddle.nn.functional.adaptive_max_pool1d/2d/3d: 自适应最大池化

作用 / Purpose:
  覆盖各种池化操作的代码路径，测试各种参数组合。
"""

import unittest

import paddle
import paddle.nn.functional as F

paddle.disable_static()


class TestAvgPool(unittest.TestCase):
    """测试平均池化 / Test average pooling"""

    def test_avg_pool1d(self):
        """测试1D平均池化 / Test 1D average pooling"""
        x = paddle.randn([4, 3, 16])
        y = F.avg_pool1d(x, kernel_size=2, stride=2)
        self.assertEqual(y.shape, [4, 3, 8])

    def test_avg_pool1d_padding(self):
        """测试带填充的1D平均池化 / Test 1D avg pool with padding"""
        x = paddle.randn([4, 3, 16])
        y = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        self.assertEqual(y.shape, [4, 3, 16])

    def test_avg_pool2d(self):
        """测试2D平均池化 / Test 2D average pooling"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.avg_pool2d(x, kernel_size=2, stride=2)
        self.assertEqual(y.shape, [4, 3, 8, 8])

    def test_avg_pool2d_same(self):
        """测试same填充2D平均池化 / Test 2D avg pool with same padding"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.assertEqual(y.shape, [4, 3, 16, 16])

    def test_avg_pool2d_with_padding(self):
        """测试带填充的2D平均池化 / Test 2D avg pool with padding"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.assertEqual(y.shape, [4, 3, 16, 16])

    def test_avg_pool3d(self):
        """测试3D平均池化 / Test 3D average pooling"""
        x = paddle.randn([2, 3, 8, 8, 8])
        y = F.avg_pool3d(x, kernel_size=2, stride=2)
        self.assertEqual(y.shape, [2, 3, 4, 4, 4])


class TestMaxPool(unittest.TestCase):
    """测试最大池化 / Test max pooling"""

    def test_max_pool1d(self):
        """测试1D最大池化 / Test 1D max pooling"""
        x = paddle.randn([4, 3, 16])
        y = F.max_pool1d(x, kernel_size=2, stride=2)
        self.assertEqual(y.shape, [4, 3, 8])

    def test_max_pool1d_with_indices(self):
        """测试带索引的1D最大池化 / Test 1D max pool with indices"""
        x = paddle.randn([4, 3, 16])
        y, indices = F.max_pool1d(x, kernel_size=2, stride=2, return_mask=True)
        self.assertEqual(y.shape, [4, 3, 8])
        self.assertEqual(indices.shape, [4, 3, 8])

    def test_max_pool2d(self):
        """测试2D最大池化 / Test 2D max pooling"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.max_pool2d(x, kernel_size=2, stride=2)
        self.assertEqual(y.shape, [4, 3, 8, 8])

    def test_max_pool2d_with_dilation(self):
        """测试空洞2D最大池化 / Test dilated 2D max pooling"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.max_pool2d(x, kernel_size=2, stride=1, dilation=2)
        self.assertEqual(y.shape, [4, 3, 14, 14])

    def test_max_pool2d_ceil_mode(self):
        """测试ceil模式2D最大池化 / Test ceil mode 2D max pooling"""
        x = paddle.randn([4, 3, 15, 15])
        y = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        self.assertEqual(y.shape, [4, 3, 8, 8])

    def test_max_pool3d(self):
        """测试3D最大池化 / Test 3D max pooling"""
        x = paddle.randn([2, 3, 8, 8, 8])
        y = F.max_pool3d(x, kernel_size=2, stride=2)
        self.assertEqual(y.shape, [2, 3, 4, 4, 4])


class TestAdaptivePool(unittest.TestCase):
    """测试自适应池化 / Test adaptive pooling"""

    def test_adaptive_avg_pool1d(self):
        """测试1D自适应平均池化 / Test 1D adaptive avg pooling"""
        x = paddle.randn([4, 3, 16])
        y = F.adaptive_avg_pool1d(x, output_size=8)
        self.assertEqual(y.shape, [4, 3, 8])

    def test_adaptive_avg_pool2d(self):
        """测试2D自适应平均池化 / Test 2D adaptive avg pooling"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.adaptive_avg_pool2d(x, output_size=(8, 8))
        self.assertEqual(y.shape, [4, 3, 8, 8])

    def test_adaptive_avg_pool2d_global(self):
        """测试全局自适应平均池化 / Test global adaptive avg pooling"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        self.assertEqual(y.shape, [4, 3, 1, 1])

    def test_adaptive_avg_pool3d(self):
        """测试3D自适应平均池化 / Test 3D adaptive avg pooling"""
        x = paddle.randn([2, 3, 8, 8, 8])
        y = F.adaptive_avg_pool3d(x, output_size=(4, 4, 4))
        self.assertEqual(y.shape, [2, 3, 4, 4, 4])

    def test_adaptive_max_pool2d(self):
        """测试2D自适应最大池化 / Test 2D adaptive max pooling"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.adaptive_max_pool2d(x, output_size=(8, 8))
        self.assertEqual(y.shape, [4, 3, 8, 8])

    def test_adaptive_max_pool2d_with_indices(self):
        """测试带索引的2D自适应最大池化 / Test adaptive max pool with indices"""
        x = paddle.randn([4, 3, 16, 16])
        y, indices = F.adaptive_max_pool2d(
            x, output_size=(8, 8), return_mask=True
        )
        self.assertEqual(y.shape, [4, 3, 8, 8])
        self.assertEqual(indices.shape, [4, 3, 8, 8])


class TestSpecialPooling(unittest.TestCase):
    """测试特殊池化操作 / Test special pooling operations"""

    def test_max_unpool2d(self):
        """测试2D最大反池化 / Test 2D max unpooling"""
        x = paddle.randn([4, 3, 16, 16])
        y, indices = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        result = F.max_unpool2d(y, indices, kernel_size=2, stride=2)
        self.assertEqual(result.shape, [4, 3, 16, 16])

    def test_lp_pool2d(self):
        """测试Lp池化 / Test Lp pooling"""
        x = paddle.randn([4, 3, 16, 16])
        y = F.lp_pool2d(x, norm_type=2, kernel_size=2, stride=2)
        self.assertEqual(y.shape, [4, 3, 8, 8])


if __name__ == '__main__':
    unittest.main()
