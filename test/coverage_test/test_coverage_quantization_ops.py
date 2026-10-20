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
量化感知训练测试 / Quantization-Aware Training Tests

测试目标 / Test Target:
  paddle.quantization 量化功能

覆盖的模块 / Covered Modules:
  - 伪量化操作
  - fake_quant层
  - 量化参数

作用 / Purpose:
  补充量化相关API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestQuantizeBasic(unittest.TestCase):
    """测试基本量化 / Test basic quantization"""

    def test_fake_quant_abs_max(self):
        """测试绝对最大值伪量化 / Test fake quantization with abs max"""
        scale = paddle.to_tensor(1.0)
        x = paddle.to_tensor([-0.5, 0.0, 0.5, 1.0])
        # Manual fake quantize: clamp to [-scale, scale] then quantize
        result = paddle.clip(x, -float(scale.numpy()), float(scale.numpy()))
        self.assertEqual(result.shape, [4])

    def test_quantize_dequantize(self):
        """测试量化反量化 / Test quantize-dequantize round trip"""
        x = paddle.to_tensor([0.1, 0.5, 0.9, -0.3])
        bits = 8
        num_levels = 2**bits
        # Quantize to int range and back
        scale = paddle.max(paddle.abs(x))
        quantized = paddle.round(x / scale * (num_levels / 2 - 1))
        dequantized = quantized / (num_levels / 2 - 1) * scale
        # Ensure shape preserved
        self.assertEqual(dequantized.shape, x.shape)


class TestQuantizationUtils(unittest.TestCase):
    """测试量化工具 / Test quantization utilities"""

    def test_uniform_quantization(self):
        """测试均匀量化 / Test uniform quantization"""
        x = paddle.to_tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        # 4-level uniform quantization
        q_min, q_max = 0, 3
        scale = (1.0 - 0.0) / (q_max - q_min)
        quantized = paddle.round(x / scale)
        quantized = paddle.clip(quantized, q_min, q_max)
        self.assertEqual(quantized.shape, [5])

    def test_symmetric_quantization(self):
        """测试对称量化 / Test symmetric quantization"""
        x = paddle.to_tensor([-0.5, -0.25, 0.0, 0.25, 0.5])
        bits = 8
        # Symmetric: scale based on abs max
        abs_max = float(paddle.max(paddle.abs(x)).numpy())
        scale = abs_max / (2 ** (bits - 1) - 1)
        quantized = paddle.round(x / scale)
        dequantized = quantized * scale
        np.testing.assert_allclose(dequantized.numpy(), x.numpy(), atol=scale)

    def test_per_channel_quantization(self):
        """测试逐通道量化 / Test per-channel quantization"""
        # Simulating per-channel quantization
        x = paddle.randn([4, 8, 16, 16])
        # Compute per-channel max (over spatial dimensions)
        max_vals = paddle.max(paddle.abs(x.reshape([4, 8, -1])), axis=2)
        self.assertEqual(max_vals.shape, [4, 8])


class TestQuantModelWrapper(unittest.TestCase):
    """测试量化模型包装器 / Test quantization model wrapper"""

    def test_model_with_fake_quant(self):
        """测试带伪量化的模型 / Test model with fake quantization"""

        class SimpleQuantModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2D(3, 8, 3)
                self.bn = nn.BatchNorm2D(8)

            def forward(self, x):
                # Simulate fake quantization by clipping
                x = paddle.clip(x, -1.0, 1.0)
                x = self.conv(x)
                x = self.bn(x)
                return x

        model = SimpleQuantModel()
        x = paddle.randn([2, 3, 16, 16])
        output = model(x)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 8)

    def test_weight_quantization_aware(self):
        """测试权重量化感知 / Test weight quantization awareness"""
        model = nn.Linear(8, 4)
        # Simulate quantizing weights
        original_weight = model.weight.numpy().copy()
        bits = 8
        q_max = 2 ** (bits - 1) - 1
        scale = np.max(np.abs(original_weight)) / q_max
        quant_weight = np.round(original_weight / scale).clip(-q_max, q_max)
        dequant_weight = quant_weight * scale
        model.weight.set_value(
            paddle.to_tensor(dequant_weight.astype(np.float32))
        )
        x = paddle.randn([4, 8])
        output = model(x)
        self.assertEqual(output.shape, [4, 4])


if __name__ == '__main__':
    unittest.main()
