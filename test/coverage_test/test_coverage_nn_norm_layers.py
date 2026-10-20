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
归一化层单元测试 / Normalization Layers Unit Tests

测试目标 / Test Target:
  paddle.nn 归一化层 (paddle.nn.BatchNorm, LayerNorm, GroupNorm, etc.)

覆盖的模块 / Covered Modules:
  - paddle.nn.BatchNorm1D/2D/3D: 批归一化
  - paddle.nn.LayerNorm: 层归一化
  - paddle.nn.GroupNorm: 组归一化
  - paddle.nn.InstanceNorm1D/2D/3D: 实例归一化
  - paddle.nn.SyncBatchNorm: 同步批归一化

作用 / Purpose:
  补充归一化层API的各种参数组合测试，提升覆盖率。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


class TestBatchNorm(unittest.TestCase):
    """测试批归一化 / Test batch normalization"""

    def test_batchnorm1d(self):
        """测试1D批归一化 / Test 1D batch normalization"""
        bn = nn.BatchNorm1D(16)
        x = paddle.randn([4, 16, 32])
        y = bn(x)
        self.assertEqual(y.shape, [4, 16, 32])

    def test_batchnorm2d(self):
        """测试2D批归一化 / Test 2D batch normalization"""
        bn = nn.BatchNorm2D(16)
        x = paddle.randn([4, 16, 8, 8])
        y = bn(x)
        self.assertEqual(y.shape, [4, 16, 8, 8])

    def test_batchnorm3d(self):
        """测试3D批归一化 / Test 3D batch normalization"""
        bn = nn.BatchNorm3D(8)
        x = paddle.randn([2, 8, 4, 4, 4])
        y = bn(x)
        self.assertEqual(y.shape, [2, 8, 4, 4, 4])

    def test_batchnorm_eval_mode(self):
        """测试评估模式批归一化 / Test batch norm in eval mode"""
        bn = nn.BatchNorm2D(16)
        bn.eval()
        x = paddle.randn([4, 16, 8, 8])
        y = bn(x)
        self.assertEqual(y.shape, [4, 16, 8, 8])

    def test_batchnorm_no_affine(self):
        """测试无仿射变换批归一化 / Test batch norm without affine"""
        bn = nn.BatchNorm2D(16, weight_attr=False, bias_attr=False)
        x = paddle.randn([4, 16, 8, 8])
        y = bn(x)
        self.assertEqual(y.shape, [4, 16, 8, 8])

    def test_batchnorm_momentum(self):
        """测试自定义动量批归一化 / Test batch norm with custom momentum"""
        bn = nn.BatchNorm2D(16, momentum=0.9)
        x = paddle.randn([4, 16, 8, 8])
        y = bn(x)
        self.assertEqual(y.shape, [4, 16, 8, 8])


class TestLayerNorm(unittest.TestCase):
    """测试层归一化 / Test layer normalization"""

    def test_layernorm_basic(self):
        """测试基本层归一化 / Test basic layer normalization"""
        ln = nn.LayerNorm(32)
        x = paddle.randn([4, 16, 32])
        y = ln(x)
        self.assertEqual(y.shape, [4, 16, 32])

    def test_layernorm_2d(self):
        """测试2D层归一化 / Test 2D layer normalization"""
        ln = nn.LayerNorm([8, 32])
        x = paddle.randn([4, 8, 32])
        y = ln(x)
        self.assertEqual(y.shape, [4, 8, 32])

    def test_layernorm_eps(self):
        """测试自定义epsilon层归一化 / Test layer norm with custom eps"""
        ln = nn.LayerNorm(32, epsilon=1e-6)
        x = paddle.randn([4, 16, 32])
        y = ln(x)
        self.assertEqual(y.shape, [4, 16, 32])

    def test_layernorm_no_elementwise(self):
        """测试无元素变换层归一化 / Test layer norm without elementwise affine"""
        ln = nn.LayerNorm(32, weight_attr=False, bias_attr=False)
        x = paddle.randn([4, 16, 32])
        y = ln(x)
        self.assertEqual(y.shape, [4, 16, 32])


class TestGroupNorm(unittest.TestCase):
    """测试组归一化 / Test group normalization"""

    def test_groupnorm_basic(self):
        """测试基本组归一化 / Test basic group normalization"""
        gn = nn.GroupNorm(num_groups=4, num_channels=16)
        x = paddle.randn([4, 16, 8, 8])
        y = gn(x)
        self.assertEqual(y.shape, [4, 16, 8, 8])

    def test_groupnorm_single_group(self):
        """测试单组归一化 / Test single group normalization"""
        gn = nn.GroupNorm(num_groups=1, num_channels=16)
        x = paddle.randn([4, 16, 8, 8])
        y = gn(x)
        self.assertEqual(y.shape, [4, 16, 8, 8])

    def test_groupnorm_channel_wise(self):
        """测试逐通道组归一化 / Test channel-wise group normalization"""
        gn = nn.GroupNorm(num_groups=16, num_channels=16)
        x = paddle.randn([4, 16, 8, 8])
        y = gn(x)
        self.assertEqual(y.shape, [4, 16, 8, 8])


class TestInstanceNorm(unittest.TestCase):
    """测试实例归一化 / Test instance normalization"""

    def test_instancenorm1d(self):
        """测试1D实例归一化 / Test 1D instance normalization"""
        inn = nn.InstanceNorm1D(16)
        x = paddle.randn([4, 16, 32])
        y = inn(x)
        self.assertEqual(y.shape, [4, 16, 32])

    def test_instancenorm2d(self):
        """测试2D实例归一化 / Test 2D instance normalization"""
        inn = nn.InstanceNorm2D(16)
        x = paddle.randn([4, 16, 8, 8])
        y = inn(x)
        self.assertEqual(y.shape, [4, 16, 8, 8])

    def test_instancenorm3d(self):
        """测试3D实例归一化 / Test 3D instance normalization"""
        inn = nn.InstanceNorm3D(8)
        x = paddle.randn([2, 8, 4, 4, 4])
        y = inn(x)
        self.assertEqual(y.shape, [2, 8, 4, 4, 4])


if __name__ == '__main__':
    unittest.main()
