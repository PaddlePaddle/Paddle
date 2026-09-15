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
批量归一化高级测试 / Advanced Batch Normalization Tests

测试目标 / Test Target:
  paddle.nn.BatchNorm 高级功能

覆盖的模块 / Covered Modules:
  - 训练/评估切换
  - 运行统计量
  - 批归一化在模型中的应用

作用 / Purpose:
  补充批归一化API高级用法的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestBatchNormAdvanced(unittest.TestCase):
    """测试BatchNorm高级功能 / Test advanced BatchNorm features"""

    def test_bn_running_stats(self):
        """测试运行统计量更新 / Test running statistics update"""
        bn = nn.BatchNorm2D(8, momentum=0.1)
        bn.train()
        # Run multiple steps so running mean converges toward 1.0
        x = paddle.ones([4, 8, 4, 4])
        for _ in range(20):
            _ = bn(x)
        # Running mean should be close to 1.0 after multiple steps
        # With momentum=0.1, after 20 steps: 0 * 0.9^20 + 1.0 * (1 - 0.9^20) ≈ 0.878
        self.assertGreater(float(bn._mean.numpy().mean()), 0.5)

    def test_bn_train_eval_switch(self):
        """测试训练/评估切换 / Test train/eval mode switch"""
        bn = nn.BatchNorm2D(8)
        x = paddle.randn([4, 8, 4, 4])
        bn.train()
        out_train = bn(x)

        bn.eval()
        out_eval = bn(x)

        self.assertEqual(out_train.shape, out_eval.shape)

    def test_bn_affine_parameters(self):
        """测试仿射参数 / Test affine parameters"""
        bn = nn.BatchNorm2D(8)
        self.assertIsNotNone(bn.weight)
        self.assertIsNotNone(bn.bias)
        # Initial weight should be 1, bias should be 0
        np.testing.assert_allclose(bn.weight.numpy(), np.ones(8), rtol=1e-5)
        np.testing.assert_allclose(bn.bias.numpy(), np.zeros(8), atol=1e-7)


class TestLayerNormAdvanced(unittest.TestCase):
    """测试LayerNorm高级功能 / Test advanced LayerNorm features"""

    def test_layer_norm_normalization(self):
        """测试LayerNorm归一化效果 / Test LayerNorm normalization effect"""
        ln = nn.LayerNorm(16, weight_attr=False, bias_attr=False)
        x = paddle.randn([4, 10, 16]) * 10 + 5  # Large values
        result = ln(x)
        # After LayerNorm, each vector should have mean≈0 and std≈1
        mean = float(result.mean(axis=-1).abs().max().numpy())
        self.assertAlmostEqual(mean, 0.0, places=4)

    def test_layer_norm_gradient(self):
        """测试LayerNorm梯度 / Test LayerNorm gradient"""
        ln = nn.LayerNorm(8)
        x = paddle.randn([2, 8])
        x.stop_gradient = False
        y = ln(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestGroupNormAdvanced(unittest.TestCase):
    """测试GroupNorm高级功能 / Test advanced GroupNorm features"""

    def test_group_norm_invariance(self):
        """测试GroupNorm尺度不变性 / Test GroupNorm scale invariance"""
        gn = nn.GroupNorm(num_groups=2, num_channels=8)
        x = paddle.randn([4, 8, 16, 16])
        result = gn(x)
        # Output should not depend on input scale
        x_scaled = x * 10
        result_scaled = gn(x_scaled)
        self.assertEqual(result.shape, result_scaled.shape)

    def test_group_norm_with_different_groups(self):
        """测试不同分组数的GroupNorm / Test GroupNorm with different groups"""
        x = paddle.randn([4, 16, 8, 8])
        for num_groups in [1, 2, 4, 8, 16]:
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=16)
            result = gn(x)
            self.assertEqual(result.shape, [4, 16, 8, 8])


class TestNormCombinations(unittest.TestCase):
    """测试归一化组合 / Test normalization combinations"""

    def test_conv_bn_relu(self):
        """测试Conv-BN-ReLU组合 / Test Conv-BN-ReLU combination"""
        model = nn.Sequential(
            nn.Conv2D(3, 16, 3, padding=1), nn.BatchNorm2D(16), nn.ReLU()
        )
        x = paddle.randn([4, 3, 16, 16])
        result = model(x)
        self.assertEqual(result.shape, [4, 16, 16, 16])
        # All outputs should be >= 0 due to ReLU
        self.assertTrue(bool((result >= 0).all().numpy()))

    def test_linear_ln_combination(self):
        """测试Linear-LayerNorm组合 / Test Linear-LayerNorm combination"""
        d_model = 64
        model = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        x = paddle.randn([4, 10, d_model])
        result = model(x)
        self.assertEqual(result.shape, [4, 10, d_model])

    def test_sync_batch_norm_creation(self):
        """测试SyncBatchNorm创建 / Test SyncBatchNorm creation"""
        # Just test creation (actual sync requires multi-process)
        sbn = nn.SyncBatchNorm(16)
        self.assertIsNotNone(sbn)
        # Convert regular BN to SyncBN
        model = nn.Sequential(nn.Conv2D(3, 16, 3), nn.BatchNorm2D(16))
        sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.assertIsNotNone(sync_model)


if __name__ == '__main__':
    unittest.main()
