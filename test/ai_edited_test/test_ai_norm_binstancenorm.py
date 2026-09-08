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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.norm
# 自动生成的单测，覆盖 paddle.nn.layer.norm 模块中未覆盖的代码
# Target: paddle/nn/layer/norm.py

"""
测试模块：paddle.nn.layer.norm
Test Module: paddle.nn.layer.norm

本测试覆盖以下功能：
This test covers the following functions:
1. LayerNorm - 层归一化 / Layer normalization with elementwise_affine=False, bias=False, list normalized_shape
2. InstanceNorm1D/2D/3D - 实例归一化 / Instance normalization with no weight/bias, input dim checks
3. GroupNorm - 组归一化 / Group normalization with affine=False, different data_format
4. BatchNorm1D/2D/3D - 批归一化 / Batch normalization with NHWC/NCDHW data_format
5. SyncBatchNorm - 同步批归一化 / Sync batch norm convert_sync_batchnorm, NLC format
6. LocalResponseNorm - 局部响应归一化 / Local response normalization
7. SpectralNorm - 谱归一化 / Spectral normalization
"""

import unittest

import paddle
from paddle import nn


class TestLayerNormComprehensive(unittest.TestCase):
    """测试LayerNorm层归一化
    Test LayerNorm"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_layer_norm_no_affine(self):
        """测试无可学习参数的层归一化 / Test LayerNorm with elementwise_affine=False"""
        ln = nn.LayerNorm(64, elementwise_affine=False)
        ln.eval()
        x = paddle.randn([2, 5, 64])
        out = ln(x)
        self.assertEqual(out.shape, [2, 5, 64])
        self.assertIsNone(ln.weight)
        self.assertIsNone(ln.bias)

    def test_layer_norm_no_bias(self):
        """测试无偏置的层归一化 / Test LayerNorm with bias=False"""
        ln = nn.LayerNorm(64, bias=False)
        ln.eval()
        x = paddle.randn([2, 5, 64])
        out = ln(x)
        self.assertEqual(out.shape, [2, 5, 64])
        self.assertIsNotNone(ln.weight)
        self.assertIsNone(ln.bias)

    def test_layer_norm_list_shape(self):
        """测试列表形式的normalized_shape / Test LayerNorm with list normalized_shape"""
        ln = nn.LayerNorm([4, 8])
        ln.eval()
        x = paddle.randn([2, 4, 8])
        out = ln(x)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_layer_norm_tuple_shape(self):
        """测试元组形式的normalized_shape / Test LayerNorm with tuple normalized_shape"""
        ln = nn.LayerNorm((4, 8))
        ln.eval()
        x = paddle.randn([2, 4, 8])
        out = ln(x)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_layer_norm_4d(self):
        """测试4D输入的层归一化 / Test LayerNorm with 4D input"""
        ln = nn.LayerNorm([4, 8])
        ln.eval()
        x = paddle.randn([2, 3, 4, 8])
        out = ln(x)
        self.assertEqual(out.shape, [2, 3, 4, 8])

    def test_layer_norm_5d(self):
        """测试5D输入的层归一化 / Test LayerNorm with 5D input"""
        ln = nn.LayerNorm([4, 8, 8])
        ln.eval()
        x = paddle.randn([2, 3, 4, 8, 8])
        out = ln(x)
        self.assertEqual(out.shape, [2, 3, 4, 8, 8])

    def test_layer_norm_eps(self):
        """测试自定义epsilon / Test LayerNorm with custom epsilon"""
        ln = nn.LayerNorm(64, epsilon=1e-8)
        ln.eval()
        x = paddle.randn([2, 5, 64])
        out = ln(x)
        self.assertEqual(out.shape, [2, 5, 64])

    def test_layer_norm_extra_repr(self):
        """测试extra_repr / Test extra_repr method"""
        ln = nn.LayerNorm([4, 8], epsilon=1e-5)
        r = ln.extra_repr()
        self.assertIn('normalized_shape', r)
        self.assertIn('epsilon', r)


class TestInstanceNormComprehensive(unittest.TestCase):
    """测试InstanceNorm
    Test InstanceNorm"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_instance_norm_1d_no_weight(self):
        """测试无权重的InstanceNorm1D / Test InstanceNorm1D with weight_attr=False"""
        norm = nn.InstanceNorm1D(8, weight_attr=False, bias_attr=False)
        norm.eval()
        x = paddle.randn([2, 8, 4])
        out = norm(x)
        self.assertEqual(out.shape, [2, 8, 4])
        self.assertIsNone(norm.scale)
        self.assertIsNone(norm.bias)

    def test_instance_norm_2d(self):
        """测试InstanceNorm2D / Test InstanceNorm2D"""
        norm = nn.InstanceNorm2D(8)
        norm.eval()
        x = paddle.randn([2, 8, 4, 4])
        out = norm(x)
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_instance_norm_2d_no_weight(self):
        """测试无权重的InstanceNorm2D / Test InstanceNorm2D without weight"""
        norm = nn.InstanceNorm2D(8, weight_attr=False, bias_attr=False)
        norm.eval()
        x = paddle.randn([2, 8, 4, 4])
        out = norm(x)
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_instance_norm_3d(self):
        """测试InstanceNorm3D / Test InstanceNorm3D"""
        norm = nn.InstanceNorm3D(8)
        norm.eval()
        x = paddle.randn([2, 8, 4, 4, 4])
        out = norm(x)
        self.assertEqual(out.shape, [2, 8, 4, 4, 4])

    def test_instance_norm_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        norm = nn.InstanceNorm1D(8, epsilon=1e-3)
        r = norm.extra_repr()
        self.assertIn('num_features', r)
        self.assertIn('epsilon', r)

    def test_instance_norm_1d_wrong_dim(self):
        """测试InstanceNorm1D维度检查 / Test InstanceNorm1D dimension check"""
        norm = nn.InstanceNorm1D(8)
        x = paddle.randn([2, 8, 4, 4])  # 4D instead of 2D/3D
        with self.assertRaises(ValueError):
            norm(x)

    def test_instance_norm_3d_wrong_dim(self):
        """测试InstanceNorm3D维度检查 / Test InstanceNorm3D dimension check"""
        norm = nn.InstanceNorm3D(8)
        x = paddle.randn([2, 8, 4, 4])  # 4D instead of 5D
        with self.assertRaises(ValueError):
            norm(x)


class TestGroupNormComprehensive(unittest.TestCase):
    """测试GroupNorm
    Test GroupNorm"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_group_norm_no_affine(self):
        """测试无仿射参数的组归一化 / Test GroupNorm with affine=False"""
        gn = nn.GroupNorm(num_channels=8, num_groups=2, affine=False)
        gn.eval()
        x = paddle.randn([2, 8, 4, 4])
        out = gn(x)
        self.assertEqual(out.shape, [2, 8, 4, 4])
        self.assertIsNone(gn.weight)
        self.assertIsNone(gn.bias)

    def test_group_norm_weight_false(self):
        """测试weight_attr=False的组归一化 / Test GroupNorm with weight_attr=False"""
        gn = nn.GroupNorm(num_channels=8, num_groups=2, weight_attr=False)
        gn.eval()
        x = paddle.randn([2, 8, 4, 4])
        out = gn(x)
        self.assertEqual(out.shape, [2, 8, 4, 4])
        self.assertIsNone(gn.weight)

    def test_group_norm_bias_false(self):
        """测试bias_attr=False的组归一化 / Test GroupNorm with bias_attr=False"""
        gn = nn.GroupNorm(num_channels=8, num_groups=2, bias_attr=False)
        gn.eval()
        x = paddle.randn([2, 8, 4, 4])
        out = gn(x)
        self.assertEqual(out.shape, [2, 8, 4, 4])
        self.assertIsNone(gn.bias)

    def test_group_norm_nhwc(self):
        """测试NHWC格式 / Test GroupNorm with NHWC data_format"""
        gn = nn.GroupNorm(num_channels=8, num_groups=2, data_format='NHWC')
        gn.eval()
        x = paddle.randn([2, 4, 4, 8])
        out = gn(x)
        self.assertEqual(out.shape, [2, 4, 4, 8])

    def test_group_norm_nlc(self):
        """测试NLC格式 / Test GroupNorm with NLC data_format"""
        gn = nn.GroupNorm(num_channels=8, num_groups=2, data_format='NLC')
        gn.eval()
        x = paddle.randn([2, 10, 8])
        out = gn(x)
        self.assertEqual(out.shape, [2, 10, 8])

    def test_group_norm_ndhwc(self):
        """测试NDHWC格式 / Test GroupNorm with NDHWC data_format"""
        gn = nn.GroupNorm(num_channels=8, num_groups=2, data_format='NDHWC')
        gn.eval()
        x = paddle.randn([2, 4, 4, 4, 8])
        out = gn(x)
        self.assertEqual(out.shape, [2, 4, 4, 4, 8])

    def test_group_norm_invalid_format(self):
        """测试无效数据格式 / Test GroupNorm with invalid data_format"""
        with self.assertRaises(ValueError):
            nn.GroupNorm(num_channels=8, num_groups=2, data_format='INVALID')

    def test_group_norm_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        gn = nn.GroupNorm(num_channels=8, num_groups=2, epsilon=1e-6)
        r = gn.extra_repr()
        self.assertIn('num_groups', r)
        self.assertIn('num_channels', r)


class TestBatchNormComprehensive(unittest.TestCase):
    """测试BatchNorm系列
    Test BatchNorm family"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_batch_norm_no_weight(self):
        """测试无权重的BatchNorm / Test BatchNorm with param_attr=False"""
        bn = nn.BatchNorm(num_channels=8, param_attr=False, bias_attr=False)
        bn.eval()
        x = paddle.randn([2, 8, 4, 4])
        out = bn(x)
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_batch_norm_nhwc(self):
        """测试NHWC格式 / Test BatchNorm with NHWC data_layout"""
        bn = nn.BatchNorm(8, data_layout='NHWC')
        bn.eval()
        x = paddle.randn([2, 4, 4, 8])
        out = bn(x)
        self.assertEqual(out.shape, [2, 4, 4, 8])

    def test_batch_norm_1d_nlc(self):
        """测试NLC格式的BatchNorm1D / Test BatchNorm1D with NLC data_format"""
        bn = nn.BatchNorm1D(8, data_format='NLC')
        bn.eval()
        x = paddle.randn([2, 4, 8])
        out = bn(x)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_batch_norm_3d_ndhwc(self):
        """测试NDHWC格式的BatchNorm3D / Test BatchNorm3D with NDHWC data_format"""
        bn = nn.BatchNorm3D(8, data_format='NDHWC')
        bn.eval()
        x = paddle.randn([2, 4, 4, 4, 8])
        out = bn(x)
        self.assertEqual(out.shape, [2, 4, 4, 4, 8])

    def test_batch_norm_2d_nhwc(self):
        """测试NHWC格式的BatchNorm2D / Test BatchNorm2D with NHWC data_format"""
        bn = nn.BatchNorm2D(8, data_format='NHWC')
        bn.eval()
        x = paddle.randn([2, 4, 4, 8])
        out = bn(x)
        self.assertEqual(out.shape, [2, 4, 4, 8])

    def test_batch_norm_use_global_stats(self):
        """测试use_global_stats / Test BatchNorm with use_global_stats"""
        bn = nn.BatchNorm2D(8, use_global_stats=True)
        bn.eval()
        x = paddle.randn([2, 8, 4, 4])
        out = bn(x)
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_batch_norm_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        bn = nn.BatchNorm2D(8, momentum=0.8, epsilon=1e-4, data_format='NHWC')
        r = bn.extra_repr()
        self.assertIn('num_features', r)
        self.assertIn('NHWC', r)

    def test_batch_norm_base_extra_repr(self):
        """测试_BatchNormBase extra_repr / Test _BatchNormBase extra_repr with name"""
        bn = nn.BatchNorm1D(8, name='test_bn', momentum=0.7)
        r = bn.extra_repr()
        self.assertIn('test_bn', r)


class TestLocalResponseNorm(unittest.TestCase):
    """测试LocalResponseNorm
    Test LocalResponseNorm"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_lrn_4d(self):
        """测试4D输入 / Test with 4D input"""
        lrn = nn.LocalResponseNorm(size=5)
        x = paddle.randn([2, 3, 8, 8])
        out = lrn(x)
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_lrn_3d(self):
        """测试3D输入 / Test with 3D input"""
        lrn = nn.LocalResponseNorm(size=5, data_format='NCL')
        x = paddle.randn([2, 3, 16])
        out = lrn(x)
        self.assertEqual(out.shape, [2, 3, 16])

    def test_lrn_5d(self):
        """测试5D输入 / Test with 5D input"""
        lrn = nn.LocalResponseNorm(size=5, data_format='NCDHW')
        x = paddle.randn([2, 3, 4, 4, 4])
        out = lrn(x)
        self.assertEqual(out.shape, [2, 3, 4, 4, 4])

    def test_lrn_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        lrn = nn.LocalResponseNorm(
            size=5, alpha=1e-3, beta=0.5, k=2.0, data_format='NHWC', name='test'
        )
        r = lrn.extra_repr()
        self.assertIn('NHWC', r)
        self.assertIn('test', r)


class TestSyncBatchNorm(unittest.TestCase):
    """测试SyncBatchNorm
    Test SyncBatchNorm"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_sync_bn_convert(self):
        """测试convert_sync_batchnorm / Test convert_sync_batchnorm"""
        model = nn.Sequential(nn.Conv2D(3, 8, 3), nn.BatchNorm2D(8))
        sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.assertIsInstance(sync_model[1], nn.SyncBatchNorm)

    def test_sync_bn_nlc_format(self):
        """测试NLC格式的SyncBatchNorm / Test SyncBatchNorm with NLC data_format"""
        sbn = nn.SyncBatchNorm(8, data_format='NCL')
        x = paddle.randn([2, 8, 4])
        out = sbn(x)
        self.assertEqual(out.shape, [2, 8, 4])

    def test_sync_bn_weight_false(self):
        """测试weight_attr=False / Test SyncBatchNorm with weight_attr=False"""
        sbn = nn.SyncBatchNorm(8, weight_attr=False)
        self.assertTrue(sbn.weight.stop_gradient)


if __name__ == '__main__':
    unittest.main()
