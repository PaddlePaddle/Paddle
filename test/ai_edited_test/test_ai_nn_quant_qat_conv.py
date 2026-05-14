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

# [AUTO-GENERATED] Test file for paddle/nn/quant/qat/conv.py
# Target file: paddle/nn/quant/qat/conv.py (92.3% coverage)
# Uncovered lines: 39 (padding_mode != 'zeros'), 65 (padding_mode != 'zeros' in forward),
#   71 (_conv_forward with non-zeros padding)

"""QAT 卷积模块测试 / Quantization-Aware Training convolution tests

测试目标 / Test Target:
  paddle/nn/quant/qat/conv.py

覆盖的模块 / Covered Modules:
  - QuantedConv2D: initialization with various configs
  - QuantedConv2D.forward: with and without quanters
  - QuantedConv2D._conv_forward: with non-zeros padding_mode
  - QuantedConv2D.weights_to_quanters / activation_quanters
  - QuantedConv2D convert via ConvertibleQuantedLayer
"""

import unittest

import paddle
from paddle import nn


class TestQuantedConv2DBasic(unittest.TestCase):
    """测试 QuantedConv2D 基本初始化和前向传播
    Test QuantedConv2D basic init and forward"""

    def setUp(self):
        paddle.disable_static()

    def test_quanted_conv2d_init_no_quanters(self):
        """测试无量化器的 QuantedConv2D 初始化
        Test QuantedConv2D init with no quanters"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3, padding=1)

        # Mock q_config with None quanters
        class MockQConfig:
            weight = None
            activation = None

        layer = QuantedConv2D(conv, MockQConfig())
        self.assertIsNotNone(layer)
        self.assertIsNone(layer.weight_quanter)
        self.assertIsNone(layer.activation_quanter)

    def test_quanted_conv2d_forward_no_quanters(self):
        """测试无量化器的前向传播
        Test forward without quanters"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3, padding=1)

        class MockQConfig:
            weight = None
            activation = None

        layer = QuantedConv2D(conv, MockQConfig())
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_quanted_conv2d_init_with_weight_quant(self):
        """测试带权重量化器的 QuantedConv2D 初始化
        Test QuantedConv2D init with weight quanter"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3, padding=1)

        class MockQuant:
            class MockInstance:
                def __call__(self, x):
                    return x

                def scales(self):
                    return paddle.to_tensor([1.0], dtype='float32')

                def zero_points(self):
                    return paddle.to_tensor([0.0], dtype='float32')

                def quant_axis(self):
                    return -1

                def bit_length(self):
                    return 8

            def _instance(self, layer):
                return self.MockInstance()

        class MockQConfig:
            weight = MockQuant()
            activation = None

        layer = QuantedConv2D(conv, MockQConfig())
        self.assertIsNotNone(layer.weight_quanter)
        self.assertIsNone(layer.activation_quanter)

    def test_quanted_conv2d_init_with_act_quant(self):
        """测试带激活量化器的 QuantedConv2D 初始化
        Test QuantedConv2D init with activation quanter"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3, padding=1)

        class MockQuant:
            class MockInstance:
                def __call__(self, x):
                    return x

            def _instance(self, layer):
                return self.MockInstance()

        class MockQConfig:
            weight = None
            activation = MockQuant()

        layer = QuantedConv2D(conv, MockQConfig())
        self.assertIsNone(layer.weight_quanter)
        self.assertIsNotNone(layer.activation_quanter)

    def test_quanted_conv2d_forward_with_both_quanters(self):
        """测试带两个量化器的前向传播
        Test forward with both quanters"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3, padding=1)

        class MockQuant:
            class MockInstance:
                def __call__(self, x):
                    return x

            def _instance(self, layer):
                return self.MockInstance()

        class MockQConfig:
            weight = MockQuant()
            activation = MockQuant()

        layer = QuantedConv2D(conv, MockQConfig())
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])


class TestQuantedConv2DNonZeroPadding(unittest.TestCase):
    """测试 QuantedConv2D 非零 padding 模式
    Test QuantedConv2D with non-zeros padding_mode"""

    def setUp(self):
        paddle.disable_static()

    def test_quanted_conv2d_reflect_padding(self):
        """测试 reflect padding 模式 (line 38-42, 64-71)
        Test reflect padding mode"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3, padding=1, padding_mode='reflect')

        class MockQConfig:
            weight = None
            activation = None

        layer = QuantedConv2D(conv, MockQConfig())
        # Should have _reversed_padding_repeated_twice
        self.assertTrue(hasattr(layer, '_reversed_padding_repeated_twice'))
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_quanted_conv2d_replicate_padding(self):
        """测试 replicate padding 模式
        Test replicate padding mode"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3, padding=1, padding_mode='replicate')

        class MockQConfig:
            weight = None
            activation = None

        layer = QuantedConv2D(conv, MockQConfig())
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])


class TestQuantedConv2DAbstractMethods(unittest.TestCase):
    """测试 QuantedConv2D 的抽象方法实现
    Test QuantedConv2D abstract method implementations"""

    def setUp(self):
        paddle.disable_static()

    def test_weights_to_quanters(self):
        """测试 weights_to_quanters 返回值
        Test weights_to_quanters return value"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3)

        class MockQConfig:
            weight = None
            activation = None

        layer = QuantedConv2D(conv, MockQConfig())
        result = layer.weights_to_quanters()
        self.assertEqual(result, [('weight', 'weight_quanter')])

    def test_activation_quanters(self):
        """测试 activation_quanters 返回值
        Test activation_quanters return value"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3)

        class MockQConfig:
            weight = None
            activation = None

        layer = QuantedConv2D(conv, MockQConfig())
        result = layer.activation_quanters()
        self.assertEqual(result, ['activation_quanter'])


class TestQuantedConv2DConvert(unittest.TestCase):
    """测试 QuantedConv2D 的转换功能
    Test QuantedConv2D conversion functionality"""

    def setUp(self):
        paddle.disable_static()

    def test_convert_with_quanters(self):
        """测试带量化器的 QuantedConv2D 转换
        Test convert QuantedConv2D with quanters"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3)

        class MockQuant:
            class MockInstance:
                def __call__(self, x):
                    return x

                def scales(self):
                    return paddle.to_tensor([1.0], dtype='float32')

                def zero_points(self):
                    return paddle.to_tensor([0.0], dtype='float32')

                def quant_axis(self):
                    return -1

                def bit_length(self):
                    return 8

            def _instance(self, layer):
                return self.MockInstance()

        class MockQConfig:
            weight = MockQuant()
            activation = MockQuant()

        layer = QuantedConv2D(conv, MockQConfig())
        self.assertFalse(layer.converted)
        layer._convert()
        self.assertTrue(layer.converted)

    def test_convert_with_remain_weight(self):
        """测试带 remain_weight 的转换
        Test convert with remain_weight=True"""
        from paddle.nn.quant.qat.conv import QuantedConv2D

        conv = nn.Conv2D(3, 16, 3)

        class MockQuant:
            class MockInstance:
                def __call__(self, x):
                    return x

                def scales(self):
                    return paddle.to_tensor([1.0], dtype='float32')

                def zero_points(self):
                    return paddle.to_tensor([0.0], dtype='float32')

                def quant_axis(self):
                    return -1

                def bit_length(self):
                    return 8

            def _instance(self, layer):
                return self.MockInstance()

        class MockQConfig:
            weight = MockQuant()
            activation = None

        layer = QuantedConv2D(conv, MockQConfig())
        layer._convert(remain_weight=True)
        self.assertTrue(layer.converted)
        # With remain_weight=True, quanter should still exist
        self.assertIsNotNone(layer.weight_quanter)


if __name__ == '__main__':
    unittest.main()
