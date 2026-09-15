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
神经网络实用工具测试 / Neural Network Utility Tests

测试目标 / Test Target:
  paddle.nn.utils 神经网络工具函数

覆盖的模块 / Covered Modules:
  - paddle.nn.utils.weight_norm: 权重归一化
  - paddle.nn.utils.spectral_norm: 谱归一化
  - paddle.nn.utils.remove_weight_norm: 移除权重归一化
  - paddle.nn.utils.parameters_to_vector: 参数转向量
  - paddle.nn.utils.vector_to_parameters: 向量转参数

作用 / Purpose:
  补充神经网络工具函数的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestWeightNorm(unittest.TestCase):
    """测试权重归一化 / Test weight normalization"""

    def test_weight_norm_linear(self):
        """测试Linear层权重归一化 / Test weight norm on Linear layer"""
        linear = nn.Linear(4, 8)
        nn.utils.weight_norm(linear)
        # After weight norm, layer has weight_g and weight_v
        self.assertTrue(hasattr(linear, 'weight_g'))
        self.assertTrue(hasattr(linear, 'weight_v'))
        x = paddle.randn([2, 4])
        y = linear(x)
        self.assertEqual(y.shape, [2, 8])

    def test_weight_norm_conv(self):
        """测试Conv层权重归一化 / Test weight norm on Conv layer"""
        conv = nn.Conv2D(3, 8, 3)
        nn.utils.weight_norm(conv)
        x = paddle.randn([2, 3, 16, 16])
        y = conv(x)
        self.assertEqual(y.shape[0], 2)

    def test_remove_weight_norm(self):
        """测试移除权重归一化 / Test remove weight norm"""
        linear = nn.Linear(4, 8)
        nn.utils.weight_norm(linear)
        nn.utils.remove_weight_norm(linear)
        # After removal, weight_g and weight_v should be gone
        self.assertFalse(hasattr(linear, 'weight_g'))


class TestSpectralNorm(unittest.TestCase):
    """测试谱归一化 / Test spectral normalization"""

    def test_spectral_norm_linear(self):
        """测试Linear层谱归一化 / Test spectral norm on Linear"""
        linear = nn.Linear(4, 8)
        nn.utils.spectral_norm(linear)
        x = paddle.randn([2, 4])
        y = linear(x)
        self.assertEqual(y.shape, [2, 8])

    def test_spectral_norm_conv(self):
        """测试Conv层谱归一化 / Test spectral norm on Conv"""
        conv = nn.Conv2D(3, 8, 3)
        nn.utils.spectral_norm(conv)
        x = paddle.randn([2, 3, 16, 16])
        y = conv(x)
        self.assertEqual(y.shape[0], 2)


class TestParameterVector(unittest.TestCase):
    """测试参数向量转换 / Test parameter vector conversion"""

    def test_parameters_to_vector(self):
        """测试参数转向量 / Test parameters to vector"""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        vec = nn.utils.parameters_to_vector(model.parameters())
        # Total params = 4*8 + 8 + 8*2 + 2 = 32+8+16+2=58
        self.assertEqual(vec.shape[0], 58)

    def test_vector_to_parameters(self):
        """测试向量转参数 / Test vector to parameters"""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        # Create a new vector of the right size
        total_params = sum(p.numel() for p in model.parameters())
        vec = paddle.zeros([total_params])
        nn.utils.vector_to_parameters(vec, model.parameters())
        # All parameters should now be zero
        for param in model.parameters():
            np.testing.assert_allclose(
                param.numpy(), np.zeros_like(param.numpy()), atol=1e-7
            )


class TestClipGrad(unittest.TestCase):
    """测试梯度裁剪 / Test gradient clipping"""

    def test_clip_grad_by_norm(self):
        """测试按范数裁剪梯度 / Test clip grad by norm"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()
        # Get grads before clipping
        grads_before = [
            p.grad.numpy().copy()
            for p in model.parameters()
            if p.grad is not None
        ]
        # Clip grads
        paddle.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        for p in model.parameters():
            if p.grad is not None:
                norm = np.linalg.norm(p.grad.numpy())
                self.assertLessEqual(
                    norm, 0.11
                )  # slightly above due to float precision

    def test_clip_grad_by_value(self):
        """测试按值裁剪梯度 / Test clip grad by value"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()
        paddle.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(bool((p.grad.abs() <= 0.5001).all().numpy()))


if __name__ == '__main__':
    unittest.main()
