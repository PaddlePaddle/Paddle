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

# [AUTO-GENERATED] Test file for paddle/nn/quant/format.py
# Target file: paddle/nn/quant/format.py (88.8% coverage)
# Uncovered lines: 31-33 (fake_fp8_quant axis>=0), 45 (fake_fp8_quant else),
#   51-53 (fake_fp8_dequant axis>=0), 59 (fake_fp8_dequant else),
#   137 (LinearQuanter float8 tuple error), 158 (LinearQuanter from_quanter),
#   218, 219 (LinearQuanter float8 tuple error in dequanter),
#   234, 298 (LinearDequanter from_quanter), 319, 375, 376, 391,
#   452, 461, 466 (ConvertibleQuantedLayer)

"""量化格式工具模块测试 / Quantization format utility tests

测试目标 / Test Target:
  paddle/nn/quant/format.py

覆盖的模块 / Covered Modules:
  - fake_fp8_quant: axis>=0, e4m3, e5m2, invalid type
  - fake_fp8_dequant: axis>=0, e4m3, e5m2, invalid type
  - LinearQuanterDequanter: forward, from_quanter
  - LinearQuanter: init with different bit_length, forward, from_quanter
  - LinearDequanter: init with different bit_length, forward, from_quanter
  - ConvertibleQuantedLayer: abstract methods, convert, quant_weights
"""

import unittest

import numpy as np

import paddle


class TestFakeFP8Quant(unittest.TestCase):
    """测试 fake_fp8_quant 函数
    Test fake_fp8_quant function"""

    def setUp(self):
        paddle.disable_static()

    def test_fake_fp8_quant_e4m3(self):
        """测试 fake_fp8_quant e4m3 类型
        Test fake_fp8_quant e4m3 type"""
        from paddle.nn.quant.format import fake_fp8_quant

        x = paddle.randn([2, 4], dtype='float32')
        scale = paddle.to_tensor([1.0], dtype='float32')
        out = fake_fp8_quant(x, scale, axis=-1, type='e4m3')
        self.assertEqual(out.shape, x.shape)

    def test_fake_fp8_quant_e5m2(self):
        """测试 fake_fp8_quant e5m2 类型
        Test fake_fp8_quant e5m2 type"""
        from paddle.nn.quant.format import fake_fp8_quant

        x = paddle.randn([2, 4], dtype='float32')
        scale = paddle.to_tensor([1.0], dtype='float32')
        out = fake_fp8_quant(x, scale, axis=-1, type='e5m2')
        self.assertEqual(out.shape, x.shape)

    def test_fake_fp8_quant_axis_positive(self):
        """测试 fake_fp8_quant 正数 axis (line 30-33)
        Test fake_fp8_quant with positive axis"""
        from paddle.nn.quant.format import fake_fp8_quant

        x = paddle.randn([2, 4, 8], dtype='float32')
        scale = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype='float32')
        out = fake_fp8_quant(x, scale, axis=1, type='e4m3')
        self.assertEqual(out.shape, x.shape)

    def test_fake_fp8_quant_invalid_type(self):
        """测试 fake_fp8_quant 无效类型 (line 45)
        Test fake_fp8_quant with invalid type"""
        from paddle.nn.quant.format import fake_fp8_quant

        x = paddle.randn([2, 4], dtype='float32')
        scale = paddle.to_tensor([1.0], dtype='float32')
        with self.assertRaises(NotImplementedError):
            fake_fp8_quant(x, scale, axis=-1, type='invalid')


class TestFakeFP8Dequant(unittest.TestCase):
    """测试 fake_fp8_dequant 函数
    Test fake_fp8_dequant function"""

    def setUp(self):
        paddle.disable_static()

    def test_fake_fp8_dequant_e4m3(self):
        """测试 fake_fp8_dequant e4m3 类型
        Test fake_fp8_dequant e4m3 type"""
        from paddle.nn.quant.format import fake_fp8_dequant

        x = paddle.randn([2, 4], dtype='float32')
        scale = paddle.to_tensor([1.0], dtype='float32')
        out = fake_fp8_dequant(x, scale, axis=-1, type='e4m3')
        self.assertEqual(out.shape, x.shape)

    def test_fake_fp8_dequant_e5m2(self):
        """测试 fake_fp8_dequant e5m2 类型
        Test fake_fp8_dequant e5m2 type"""
        from paddle.nn.quant.format import fake_fp8_dequant

        x = paddle.randn([2, 4], dtype='float32')
        scale = paddle.to_tensor([1.0], dtype='float32')
        out = fake_fp8_dequant(x, scale, axis=-1, type='e5m2')
        self.assertEqual(out.shape, x.shape)

    def test_fake_fp8_dequant_axis_positive(self):
        """测试 fake_fp8_dequant 正数 axis (line 50-53)
        Test fake_fp8_dequant with positive axis"""
        from paddle.nn.quant.format import fake_fp8_dequant

        x = paddle.randn([2, 4, 8], dtype='float32')
        scale = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype='float32')
        out = fake_fp8_dequant(x, scale, axis=1, type='e4m3')
        self.assertEqual(out.shape, x.shape)

    def test_fake_fp8_dequant_invalid_type(self):
        """测试 fake_fp8_dequant 无效类型 (line 59)
        Test fake_fp8_dequant with invalid type"""
        from paddle.nn.quant.format import fake_fp8_dequant

        x = paddle.randn([2, 4], dtype='float32')
        scale = paddle.to_tensor([1.0], dtype='float32')
        with self.assertRaises(NotImplementedError):
            fake_fp8_dequant(x, scale, axis=-1, type='invalid')


class TestLinearQuanterDequanter(unittest.TestCase):
    """测试 LinearQuanterDequanter 类
    Test LinearQuanterDequanter class"""

    def setUp(self):
        paddle.disable_static()

    def test_linear_qdq_both_none(self):
        """测试 quanter 和 dequanter 都为 None
        Test with both quanter and dequanter as None"""
        from paddle.nn.quant.format import LinearQuanterDequanter

        layer = LinearQuanterDequanter(quanter=None, dequanter=None)
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_linear_qdq_from_quanter(self):
        """测试 LinearQuanterDequanter.from_quanter (line 76-82)
        Test LinearQuanterDequanter.from_quanter"""
        from paddle.nn.quant.format import LinearQuanterDequanter

        # Create a mock quanter with required methods
        class MockQuanter:
            def scales(self):
                return paddle.to_tensor([1.0], dtype='float32')

            def zero_points(self):
                return paddle.to_tensor([0.0], dtype='float32')

            def quant_axis(self):
                return -1

            def bit_length(self):
                return 8

        qdq = LinearQuanterDequanter.from_quanter(MockQuanter())
        self.assertIsNotNone(qdq._quanter)
        self.assertIsNotNone(qdq._dequanter)


class TestLinearQuanter(unittest.TestCase):
    """测试 LinearQuanter 类
    Test LinearQuanter class"""

    def setUp(self):
        paddle.disable_static()

    def test_linear_quanter_basic(self):
        """测试 LinearQuanter 基本量化
        Test LinearQuanter basic quantization"""
        from paddle.nn.quant.format import LinearQuanter

        layer = LinearQuanter(
            scales=[1.0],
            zero_point=0.0,
            quant_axis=-1,
            bit_length=8,
        )
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_linear_quanter_with_zero_point(self):
        """测试 LinearQuanter 带 zero_point (line 157-177)
        Test LinearQuanter with non-zero zero_point"""
        from paddle.nn.quant.format import LinearQuanter

        # Use multi-dim scales to hit the zero_point path
        scales = paddle.randn([2, 4], dtype='float32').abs() + 0.1
        zp = paddle.ones([2, 4], dtype='float32') * 5.0
        layer = LinearQuanter(
            scales=scales,
            zero_point=zp,
            quant_axis=0,
            bit_length=8,
        )
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_linear_quanter_e4m3_tuple(self):
        """测试 LinearQuanter float8 e4m3 tuple bit_length
        Test LinearQuanter with float8 e4m3 tuple bit_length"""
        from paddle.nn.quant.format import LinearQuanter

        layer = LinearQuanter(
            scales=[1.0],
            zero_point=0.0,
            quant_axis=-1,
            bit_length=(4, 3),
        )
        self.assertEqual(layer._qmax, 448)
        self.assertEqual(layer._qmin, -448)

        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_linear_quanter_e5m2_tuple(self):
        """测试 LinearQuanter float8 e5m2 tuple bit_length
        Test LinearQuanter with float8 e5m2 tuple bit_length"""
        from paddle.nn.quant.format import LinearQuanter

        layer = LinearQuanter(
            scales=[1.0],
            zero_point=0.0,
            quant_axis=-1,
            bit_length=(5, 2),
        )
        self.assertEqual(layer._qmax, 57344)
        self.assertEqual(layer._qmin, -57344)

        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_linear_quanter_invalid_tuple(self):
        """测试 LinearQuanter 无效 tuple bit_length (line 137)
        Test LinearQuanter with invalid tuple bit_length"""
        from paddle.nn.quant.format import LinearQuanter

        with self.assertRaises(NotImplementedError):
            LinearQuanter(
                scales=[1.0],
                zero_point=0.0,
                bit_length=(3, 3),
            )

    def test_linear_quanter_from_quanter(self):
        """测试 LinearQuanter.from_quanter 静态方法
        Test LinearQuanter.from_quanter static method"""
        from paddle.nn.quant.format import LinearQuanter

        class MockQuanter:
            def scales(self):
                return paddle.to_tensor([1.0], dtype='float32')

            def zero_points(self):
                return paddle.to_tensor([0.0], dtype='float32')

            def quant_axis(self):
                return -1

            def bit_length(self):
                return 8

        layer = LinearQuanter.from_quanter(MockQuanter())
        self.assertIsNotNone(layer)


class TestLinearDequanter(unittest.TestCase):
    """测试 LinearDequanter 类
    Test LinearDequanter class"""

    def setUp(self):
        paddle.disable_static()

    def test_linear_dequanter_basic(self):
        """测试 LinearDequanter 基本反量化
        Test LinearDequanter basic dequantization"""
        from paddle.nn.quant.format import LinearDequanter

        layer = LinearDequanter(
            scales=[1.0],
            zero_point=0.0,
            quant_axis=-1,
            bit_length=8,
        )
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_linear_dequanter_with_zero_point(self):
        """测试 LinearDequanter 带 zero_point (line 318-322)
        Test LinearDequanter with non-zero zero_point"""
        from paddle.nn.quant.format import LinearDequanter

        scales = paddle.randn([2, 4], dtype='float32').abs() + 0.1
        zp = paddle.ones([2, 4], dtype='float32') * 5.0
        layer = LinearDequanter(
            scales=scales,
            zero_point=zp,
            quant_axis=0,
            bit_length=8,
        )
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_linear_dequanter_e4m3(self):
        """测试 LinearDequanter float8 e4m3
        Test LinearDequanter float8 e4m3"""
        from paddle.nn.quant.format import LinearDequanter

        layer = LinearDequanter(
            scales=[1.0],
            zero_point=0.0,
            quant_axis=-1,
            bit_length=(4, 3),
        )
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_linear_dequanter_e5m2(self):
        """测试 LinearDequanter float8 e5m2
        Test LinearDequanter float8 e5m2"""
        from paddle.nn.quant.format import LinearDequanter

        layer = LinearDequanter(
            scales=[1.0],
            zero_point=0.0,
            quant_axis=-1,
            bit_length=(5, 2),
        )
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_linear_dequanter_from_quanter(self):
        """测试 LinearDequanter.from_quanter 静态方法
        Test LinearDequanter.from_quanter static method"""
        from paddle.nn.quant.format import LinearDequanter

        class MockQuanter:
            def scales(self):
                return paddle.to_tensor([1.0], dtype='float32')

            def zero_points(self):
                return paddle.to_tensor([0.0], dtype='float32')

            def quant_axis(self):
                return -1

            def bit_length(self):
                return 8

        layer = LinearDequanter.from_quanter(MockQuanter())
        self.assertIsNotNone(layer)


class TestConvertibleQuantedLayer(unittest.TestCase):
    """测试 ConvertibleQuantedLayer 抽象类
    Test ConvertibleQuantedLayer abstract class"""

    def setUp(self):
        paddle.disable_static()

    def test_convertible_quanted_layer_convert(self):
        """测试 ConvertibleQuantedLayer._convert (line 481-494)
        Test ConvertibleQuantedLayer._convert"""
        from paddle.nn.quant.format import ConvertibleQuantedLayer

        class DummyQuanter:
            def scales(self):
                return paddle.to_tensor([1.0], dtype='float32')

            def zero_points(self):
                return paddle.to_tensor([0.0], dtype='float32')

            def quant_axis(self):
                return -1

            def bit_length(self):
                return 8

        class ConcreteQuantedLayer(ConvertibleQuantedLayer):
            def __init__(self):
                super().__init__()
                self.weight = paddle.create_parameter(
                    shape=[4, 8], dtype='float32'
                )
                self.weight_quanter = DummyQuanter()
                self.activation_quanter = DummyQuanter()

            def forward(self, x):
                return x

            def weights_to_quanters(self):
                return [('weight', 'weight_quanter')]

            def activation_quanters(self):
                return ['activation_quanter']

        layer = ConcreteQuantedLayer()
        self.assertFalse(layer.converted)
        layer._convert()
        self.assertTrue(layer.converted)

    def test_convertible_quanted_layer_convert_remain_weight(self):
        """测试 ConvertibleQuantedLayer._convert 带 remain_weight
        Test ConvertibleQuantedLayer._convert with remain_weight=True"""
        from paddle.nn.quant.format import ConvertibleQuantedLayer

        class DummyQuanter:
            def scales(self):
                return paddle.to_tensor([1.0], dtype='float32')

            def zero_points(self):
                return paddle.to_tensor([0.0], dtype='float32')

            def quant_axis(self):
                return -1

            def bit_length(self):
                return 8

        class ConcreteLayer(ConvertibleQuantedLayer):
            def __init__(self):
                super().__init__()
                self.weight = paddle.create_parameter(
                    shape=[4, 8], dtype='float32'
                )
                self.weight_quanter = DummyQuanter()
                self.activation_quanter = DummyQuanter()

            def forward(self, x):
                return x

            def weights_to_quanters(self):
                return [('weight', 'weight_quanter')]

            def activation_quanters(self):
                return ['activation_quanter']

        layer = ConcreteLayer()
        layer._convert(remain_weight=True)
        self.assertTrue(layer.converted)
        # With remain_weight, quanter should still exist
        self.assertIsNotNone(layer.weight_quanter)

    def test_convert_quanter_to_qdq_no_attr(self):
        """测试 _convert_quanter_to_qdq 属性不存在 (line 465-466)
        Test _convert_quanter_to_qdq when attribute doesn't exist"""
        from paddle.nn.quant.format import ConvertibleQuantedLayer

        class ConcreteLayer(ConvertibleQuantedLayer):
            def __init__(self):
                super().__init__()
                # Don't add quanter attribute

            def forward(self, x):
                return x

            def weights_to_quanters(self):
                return []

            def activation_quanters(self):
                return ['nonexistent_quanter']

        layer = ConcreteLayer()
        result = layer._convert_quanter_to_qdq('nonexistent_quanter')
        self.assertIsNone(result)

    def test_convert_twice_raises(self):
        """测试重复转换抛出异常 (line 483)
        Test converting twice raises assertion error"""
        from paddle.nn.quant.format import ConvertibleQuantedLayer

        class ConcreteLayer(ConvertibleQuantedLayer):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

            def weights_to_quanters(self):
                return []

            def activation_quanters(self):
                return []

        layer = ConcreteLayer()
        layer._convert()
        with self.assertRaises(AssertionError):
            layer._convert()

    def test_quant_weights(self):
        """测试 _quant_weights (line 475-479)
        Test _quant_weights"""
        from paddle.nn.quant.format import (
            ConvertibleQuantedLayer,
            LinearQuanter,
        )

        class ConcreteLayer(ConvertibleQuantedLayer):
            def __init__(self):
                super().__init__()
                self.weight = paddle.create_parameter(
                    shape=[4, 8], dtype='float32'
                )

            def forward(self, x):
                return x

            def weights_to_quanters(self):
                return []

            def activation_quanters(self):
                return []

        layer = ConcreteLayer()
        quanter = LinearQuanter(scales=[1.0], zero_point=0.0, bit_length=8)
        layer._quant_weights('weight', quanter)
        # Weight should still exist after quantization
        self.assertIsNotNone(layer.weight)


if __name__ == '__main__':
    unittest.main()
