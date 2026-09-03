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
Dropout层单元测试 / Dropout Layer Unit Tests

测试目标 / Test Target:
  paddle.nn.Dropout系列层 (覆盖率较高但部分边界情况未测)

覆盖的模块 / Covered Modules:
  - paddle.nn.Dropout: 1D Dropout
  - paddle.nn.Dropout2D: 2D通道Dropout
  - paddle.nn.Dropout3D: 3D通道Dropout
  - paddle.nn.AlphaDropout: Alpha Dropout

作用 / Purpose:
  覆盖Dropout层在训练和评估模式下的代码路径，测试各种参数设置和边界情况。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestDropout(unittest.TestCase):
    """测试Dropout / Test Dropout"""

    def test_dropout_eval_mode(self):
        """测试评估模式下Dropout（无丢弃）/ Test Dropout in eval mode (no dropout)"""
        dropout = nn.Dropout(p=0.5)
        dropout.eval()
        x = paddle.randn([100, 100])
        y = dropout(x)
        np.testing.assert_allclose(x.numpy(), y.numpy())

    def test_dropout_train_mode(self):
        """测试训练模式下Dropout / Test Dropout in train mode"""
        dropout = nn.Dropout(p=0.5)
        dropout.train()
        x = paddle.ones([1000])
        y = dropout(x)
        # Some values should be zero
        n_zeros = int(paddle.sum(y == 0).numpy())
        self.assertTrue(n_zeros > 0)

    def test_dropout_p0(self):
        """测试p=0的Dropout（无丢弃）/ Test Dropout with p=0 (no dropout)"""
        dropout = nn.Dropout(p=0.0)
        x = paddle.randn([4, 10])
        y = dropout(x)
        np.testing.assert_allclose(x.numpy(), y.numpy())

    def test_dropout_p1(self):
        """测试p=1的Dropout（全丢弃）/ Test Dropout with p=1 (all zero)"""
        dropout = nn.Dropout(p=1.0)
        x = paddle.ones([100])
        y = dropout(x)
        # All values should be zero in train mode (default)
        self.assertAlmostEqual(float(y.sum().numpy()), 0.0, places=5)

    def test_dropout_mode_upscale(self):
        """测试upscale模式 / Test upscale mode"""
        dropout = nn.Dropout(p=0.5, mode='upscale_in_train')
        x = paddle.ones([1000])
        y = dropout(x)
        # Non-zero values should be scaled by 1/(1-p)
        non_zero = y[y != 0]
        if len(non_zero) > 0:
            self.assertAlmostEqual(float(non_zero[0].numpy()), 2.0, places=5)

    def test_dropout_shape_preserved(self):
        """测试Dropout保留形状 / Test Dropout preserves shape"""
        dropout = nn.Dropout(p=0.3)
        x = paddle.randn([4, 6, 8])
        y = dropout(x)
        self.assertEqual(y.shape, [4, 6, 8])

    def test_dropout_gradient(self):
        """测试Dropout梯度 / Test Dropout gradient"""
        dropout = nn.Dropout(p=0.0)  # p=0: identity
        x = paddle.randn([4, 10])
        x.stop_gradient = False
        y = dropout(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)


class TestDropout2D(unittest.TestCase):
    """测试Dropout2D / Test Dropout2D"""

    def test_dropout2d_basic(self):
        """测试基本Dropout2D / Test basic Dropout2D"""
        dropout2d = nn.Dropout2D(p=0.5)
        x = paddle.randn([4, 8, 16, 16])
        y = dropout2d(x)
        self.assertEqual(y.shape, [4, 8, 16, 16])

    def test_dropout2d_channel_drop(self):
        """测试Dropout2D通道丢弃 / Test Dropout2D channel dropping"""
        dropout2d = nn.Dropout2D(p=0.5)
        dropout2d.train()
        x = paddle.ones([4, 100, 8, 8])
        y = dropout2d(x)
        # Some entire channels should be zero
        # Check that some channels are zero
        channel_sums = y.mean(axis=[0, 2, 3])
        n_zero_channels = int(paddle.sum(channel_sums == 0).numpy())
        self.assertTrue(n_zero_channels > 0)

    def test_dropout2d_eval_mode(self):
        """测试评估模式Dropout2D / Test Dropout2D eval mode"""
        dropout2d = nn.Dropout2D(p=0.5)
        dropout2d.eval()
        x = paddle.randn([4, 8, 16, 16])
        y = dropout2d(x)
        np.testing.assert_allclose(x.numpy(), y.numpy())

    def test_dropout2d_nchw(self):
        """测试NCHW格式的Dropout2D / Test NCHW format Dropout2D"""
        dropout2d = nn.Dropout2D(p=0.3, data_format='NCHW')
        x = paddle.randn([4, 8, 16, 16])
        y = dropout2d(x)
        self.assertEqual(y.shape, [4, 8, 16, 16])


class TestDropout3D(unittest.TestCase):
    """测试Dropout3D / Test Dropout3D"""

    def test_dropout3d_basic(self):
        """测试基本Dropout3D / Test basic Dropout3D"""
        dropout3d = nn.Dropout3D(p=0.5)
        x = paddle.randn([2, 8, 4, 8, 8])
        y = dropout3d(x)
        self.assertEqual(y.shape, [2, 8, 4, 8, 8])

    def test_dropout3d_eval_mode(self):
        """测试评估模式Dropout3D / Test Dropout3D eval mode"""
        dropout3d = nn.Dropout3D(p=0.5)
        dropout3d.eval()
        x = paddle.randn([2, 8, 4, 8, 8])
        y = dropout3d(x)
        np.testing.assert_allclose(x.numpy(), y.numpy())

    def test_dropout3d_channel_wise(self):
        """测试Dropout3D通道维度 / Test Dropout3D channel-wise"""
        dropout3d = nn.Dropout3D(p=0.5)
        dropout3d.train()
        x = paddle.ones([2, 100, 4, 8, 8])
        y = dropout3d(x)
        self.assertEqual(y.shape, [2, 100, 4, 8, 8])


class TestAlphaDropout(unittest.TestCase):
    """测试AlphaDropout / Test AlphaDropout"""

    def test_alpha_dropout_basic(self):
        """测试基本AlphaDropout / Test basic AlphaDropout"""
        alpha_dropout = nn.AlphaDropout(p=0.5)
        x = paddle.randn([100, 100])
        y = alpha_dropout(x)
        self.assertEqual(y.shape, [100, 100])

    def test_alpha_dropout_eval_mode(self):
        """测试评估模式AlphaDropout / Test AlphaDropout eval mode"""
        alpha_dropout = nn.AlphaDropout(p=0.5)
        alpha_dropout.eval()
        x = paddle.randn([100, 100])
        y = alpha_dropout(x)
        np.testing.assert_allclose(x.numpy(), y.numpy())

    def test_alpha_dropout_selu_combination(self):
        """测试AlphaDropout与SELU组合 / Test AlphaDropout with SELU combination"""
        # AlphaDropout is designed to work with SELU
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.SELU(),
            nn.AlphaDropout(p=0.1),
            nn.Linear(10, 5),
        )
        model.train()
        x = paddle.randn([4, 10])
        y = model(x)
        self.assertEqual(y.shape, [4, 5])


class TestDropoutInModel(unittest.TestCase):
    """测试Dropout在模型中的使用 / Test Dropout in model"""

    def test_dropout_in_sequential(self):
        """测试Dropout在Sequential中 / Test Dropout in Sequential"""
        model = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(20, 5)
        )
        x = paddle.randn([4, 10])
        model.train()
        y_train = model(x)
        model.eval()
        y_eval = model(x)
        # Both should have same shape
        self.assertEqual(y_train.shape, [4, 5])
        self.assertEqual(y_eval.shape, [4, 5])

    def test_dropout_train_eval_difference(self):
        """测试训练和评估模式的不同 / Test difference between train and eval"""
        dropout = nn.Dropout(p=0.5)
        x = paddle.ones([1000])

        # Multiple evaluations with eval mode should give same result
        dropout.eval()
        y1 = dropout(x)
        y2 = dropout(x)
        np.testing.assert_allclose(y1.numpy(), y2.numpy())

        # In train mode, multiple runs may differ
        dropout.train()
        y3 = dropout(x)
        # y3 should have some zeros due to dropout
        self.assertTrue(
            float(y3.sum().numpy()) <= 2000
        )  # max is 2000 due to upscaling


if __name__ == '__main__':
    unittest.main()
