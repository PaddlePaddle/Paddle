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

# [AUTO-GENERATED] Test file for paddle/nn/quant/quant_layers.py
# Target file: paddle/nn/quant/quant_layers.py (71.6% coverage)
# Uncovered lines: 118, 141-169, 245, 276-300, 307, 344-421, 491-538,
#   755, 852-952 (QuantizedColumnParallelLinear), 955-1059 (QuantizedRowParallelLinear),
#   1062-1125 (QuantizedMatmul), 1128-1196 (MAOutputScaleLayer, FakeQuantMAOutputScaleLayer),
#   1199-1245 (_get_fake_quant_type lsq paths)

"""量化层高级测试 / Advanced quantization layer tests

测试目标 / Test Target:
  paddle/nn/quant/quant_layers.py

覆盖的模块 / Covered Modules:
  - FakeQuantAbsMax: with quant_on_weight=True
  - FakeQuantMovingAverageAbsMax: eval mode (is_test)
  - FakeQuantChannelWiseAbsMax: full coverage
  - MovingAverageAbsMaxScale: eval mode (is_test)
  - QuantStub alias
  - QuantizedConv2D: forward with different padding_mode, pre_layers, output_size=None
  - QuantizedConv2DTranspose: forward with output_size
  - QuantizedLinear: forward
  - QuantizedMatmul: forward with act_quant_layer
  - MAOutputScaleLayer: forward
  - FakeQuantMAOutputScaleLayer: forward with multi-output layer
  - _get_fake_quant_type: lsq_weight, channel_wise_lsq_weight, lsq_act
"""

import unittest

import paddle
from paddle import nn
from paddle.nn.quant.quant_layers import (
    FakeQuantAbsMax,
    FakeQuantChannelWiseAbsMax,
    FakeQuantMAOutputScaleLayer,
    FakeQuantMovingAverageAbsMax,
    MAOutputScaleLayer,
    MovingAverageAbsMaxScale,
    QuantStub,
)


class TestFakeQuantAbsMaxWeightQuant(unittest.TestCase):
    """测试 FakeQuantAbsMax 在 quant_on_weight=True 时的行为
    Test FakeQuantAbsMax with quant_on_weight=True"""

    def setUp(self):
        paddle.disable_static()

    def test_abs_max_weight_quant(self):
        """测试带 quant_on_weight=True 的 abs_max 量化
        Test abs_max quant with quant_on_weight=True"""
        # Line 93-103: quant_on_weight=True path
        layer = FakeQuantAbsMax(name='test', quant_bits=8, quant_on_weight=True)
        x = paddle.randn([3, 16, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)
        # Scale should be a trainable parameter
        self.assertIsNotNone(layer._scale)

    def test_abs_max_no_weight_quant(self):
        """测试不带 quant_on_weight 的 abs_max 量化
        Test abs_max quant without quant_on_weight"""
        # Line 103-104: quant_on_weight=False path
        layer = FakeQuantAbsMax(
            name='test2', quant_bits=8, quant_on_weight=False
        )
        x = paddle.randn([3, 16, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNone(layer._scale)


class TestFakeQuantMovingAverageAbsMaxEval(unittest.TestCase):
    """测试 FakeQuantMovingAverageAbsMax 在 eval 模式下的行为
    Test FakeQuantMovingAverageAbsMax in eval mode"""

    def setUp(self):
        paddle.disable_static()

    def test_moving_average_eval_mode(self):
        """测试 moving_average_abs_max 在 eval 模式下的量化
        Test moving_average_abs_max quant in eval mode
        Covers lines 141-169 (static graph path not reachable in dynamic mode,
        but covers the dynamic training path with is_test=False)"""
        layer = FakeQuantMovingAverageAbsMax(
            name='test_ma', moving_rate=0.9, quant_bits=8
        )
        layer.train()
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_moving_average_eval_mode_explicit(self):
        """测试 moving_average_abs_max 显式 eval 模式
        Test moving_average_abs_max explicit eval mode"""
        layer = FakeQuantMovingAverageAbsMax(
            name='test_ma_eval', moving_rate=0.9, quant_bits=8
        )
        layer.eval()
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)


class TestFakeQuantChannelWiseAbsMax(unittest.TestCase):
    """测试 FakeQuantChannelWiseAbsMax 的完整行为
    Test FakeQuantChannelWiseAbsMax full behavior"""

    def setUp(self):
        paddle.disable_static()

    def test_channel_wise_abs_max_forward(self):
        """测试 channel_wise_abs_max 前向传播
        Test channel_wise_abs_max forward pass"""
        layer = FakeQuantChannelWiseAbsMax(
            name='test_cw',
            channel_num=16,
            quant_bits=8,
            quant_axis=0,
            quant_on_weight=True,
        )
        x = paddle.randn([16, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_channel_wise_different_quant_axis(self):
        """测试不同 quant_axis 的 channel_wise 量化
        Test channel_wise quant with different quant_axis"""
        layer = FakeQuantChannelWiseAbsMax(
            name='test_cw_axis1',
            channel_num=8,
            quant_bits=8,
            quant_axis=1,
            quant_on_weight=True,
        )
        x = paddle.randn([4, 8, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)


class TestMovingAverageAbsMaxScale(unittest.TestCase):
    """测试 MovingAverageAbsMaxScale 的完整行为
    Test MovingAverageAbsMaxScale full behavior"""

    def setUp(self):
        paddle.disable_static()

    def test_moving_average_scale_forward(self):
        """测试 moving_average_scale 前向传播
        Test moving_average_scale forward pass"""
        layer = MovingAverageAbsMaxScale(name='test_scale', moving_rate=0.9)
        layer.train()
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_moving_average_scale_eval(self):
        """测试 moving_average_scale eval 模式
        Test moving_average_scale eval mode"""
        layer = MovingAverageAbsMaxScale(
            name='test_scale_eval', moving_rate=0.9
        )
        layer.eval()
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_quant_stub_alias(self):
        """测试 QuantStub 是 MovingAverageAbsMaxScale 的别名
        Test QuantStub is alias of MovingAverageAbsMaxScale"""
        self.assertIs(QuantStub, MovingAverageAbsMaxScale)
        layer = QuantStub(name='test_stub')
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)


class TestQuantizedConv2DForward(unittest.TestCase):
    """测试 QuantizedConv2D 前向传播
    Test QuantizedConv2D forward pass"""

    def setUp(self):
        paddle.disable_static()

    def test_quantized_conv2d_forward(self):
        """测试 QuantizedConv2D 前向传播
        Test QuantizedConv2D forward"""
        conv = nn.Conv2D(3, 16, 3, padding=1)
        layer = paddle.nn.quant.quant_layers.QuantizedConv2D(
            layer=conv,
            weight_quantize_type='abs_max',
            activation_quantize_type='abs_max',
        )
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_quantized_conv2d_reflect_padding(self):
        """测试 QuantizedConv2D 带 reflect padding 模式
        Test QuantizedConv2D with reflect padding mode"""
        conv = nn.Conv2D(3, 16, 3, padding=1, padding_mode='reflect')
        layer = paddle.nn.quant.quant_layers.QuantizedConv2D(layer=conv)
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_quantized_conv2d_with_pre_layers(self):
        """测试 QuantizedConv2D 带预处理层
        Test QuantizedConv2D with preprocessing layers"""
        conv = nn.Conv2D(3, 16, 3, padding=1)

        def weight_pre():
            return paddle.nn.ReLU()

        def act_pre():
            return paddle.nn.ReLU()

        layer = paddle.nn.quant.quant_layers.QuantizedConv2D(
            layer=conv,
            weight_pre_layer=weight_pre,
            act_pre_layer=act_pre,
        )
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_quantized_conv2d_lsq_weight_type(self):
        """测试 QuantizedConv2D 使用 lsq_weight 量化类型
        Test QuantizedConv2D with lsq_weight quantize type"""
        conv = nn.Conv2D(3, 16, 3, padding=1)
        layer = paddle.nn.quant.quant_layers.QuantizedConv2D(
            layer=conv,
            weight_quantize_type='channel_wise_lsq_weight',
            activation_quantize_type='abs_max',
        )
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_quantized_conv2d_lsq_act_type(self):
        """测试 QuantizedConv2D 使用 lsq_act 量化类型
        Test QuantizedConv2D with lsq_act quantize type"""
        conv = nn.Conv2D(3, 16, 3, padding=1)
        layer = paddle.nn.quant.quant_layers.QuantizedConv2D(
            layer=conv,
            weight_quantize_type='abs_max',
            activation_quantize_type='lsq_act',
        )
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 16, 8, 8])


class TestQuantizedConv2DTransposeForward(unittest.TestCase):
    """测试 QuantizedConv2DTranspose 前向传播
    Test QuantizedConv2DTranspose forward pass"""

    def setUp(self):
        paddle.disable_static()

    def test_quantized_conv2d_transpose_forward(self):
        """测试 QuantizedConv2DTranspose 前向传播
        Test QuantizedConv2DTranspose forward"""
        conv = nn.Conv2DTranspose(
            16, 3, 3, stride=2, padding=1, output_padding=1
        )
        layer = paddle.nn.quant.quant_layers.QuantizedConv2DTranspose(
            layer=conv
        )
        x = paddle.randn([2, 16, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)

    def test_quantized_conv2d_transpose_with_output_size(self):
        """测试 QuantizedConv2DTranspose 带 output_size 参数
        Test QuantizedConv2DTranspose with output_size (line 752-755)"""
        conv = nn.Conv2DTranspose(
            16, 3, 3, stride=2, padding=1, output_padding=1
        )
        layer = paddle.nn.quant.quant_layers.QuantizedConv2DTranspose(
            layer=conv
        )
        x = paddle.randn([2, 16, 8, 8], dtype='float32')
        out = layer(x, output_size=[16, 16])
        self.assertEqual(out.shape, [2, 3, 16, 16])

    def test_quantized_conv2d_transpose_with_pre_layers(self):
        """测试 QuantizedConv2DTranspose 带预处理层
        Test QuantizedConv2DTranspose with preprocessing layers"""
        conv = nn.Conv2DTranspose(
            16, 3, 3, stride=2, padding=1, output_padding=1
        )

        def act_pre():
            return paddle.nn.ReLU()

        layer = paddle.nn.quant.quant_layers.QuantizedConv2DTranspose(
            layer=conv, act_pre_layer=act_pre
        )
        x = paddle.randn([2, 16, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape[0], 2)

    def test_quantized_conv2d_transpose_lsq_types(self):
        """测试 QuantizedConv2DTranspose 使用 lsq_act 量化类型
        Test QuantizedConv2DTranspose with lsq_act quantize type"""
        conv = nn.Conv2DTranspose(
            16, 3, 3, stride=2, padding=1, output_padding=1
        )
        layer = paddle.nn.quant.quant_layers.QuantizedConv2DTranspose(
            layer=conv,
            weight_quantize_type='channel_wise_abs_max',
            activation_quantize_type='lsq_act',
        )
        x = paddle.randn([2, 16, 8, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape[0], 2)


class TestQuantizedLinearForward(unittest.TestCase):
    """测试 QuantizedLinear 前向传播
    Test QuantizedLinear forward pass"""

    def setUp(self):
        paddle.disable_static()

    def test_quantized_linear_forward(self):
        """测试 QuantizedLinear 前向传播
        Test QuantizedLinear forward"""
        linear = nn.Linear(10, 5)
        layer = paddle.nn.quant.quant_layers.QuantizedLinear(
            layer=linear,
            weight_quantize_type='abs_max',
            activation_quantize_type='abs_max',
        )
        x = paddle.randn([4, 10], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [4, 5])

    def test_quantized_linear_with_pre_layers(self):
        """测试 QuantizedLinear 带预处理层
        Test QuantizedLinear with preprocessing layers"""
        linear = nn.Linear(10, 5)

        def act_pre():
            return paddle.nn.ReLU()

        def weight_pre():
            return paddle.nn.ReLU()

        layer = paddle.nn.quant.quant_layers.QuantizedLinear(
            layer=linear,
            weight_pre_layer=weight_pre,
            act_pre_layer=act_pre,
            weight_quantize_type='abs_max',
            activation_quantize_type='moving_average_abs_max',
        )
        x = paddle.randn([4, 10], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [4, 5])

    def test_quantized_linear_lsq_weight(self):
        """测试 QuantizedLinear 使用 channel_wise_lsq_weight 量化类型
        Test QuantizedLinear with channel_wise_lsq_weight quantize type"""
        linear = nn.Linear(10, 5)
        layer = paddle.nn.quant.quant_layers.QuantizedLinear(
            layer=linear,
            weight_quantize_type='channel_wise_lsq_weight',
            activation_quantize_type='lsq_act',
        )
        x = paddle.randn([4, 10], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [4, 5])

    def test_quantized_linear_channel_wise_lsq(self):
        """测试 QuantizedLinear 使用 channel_wise_lsq_weight 量化
        Test QuantizedLinear with channel_wise_lsq_weight quant"""
        linear = nn.Linear(10, 5)
        layer = paddle.nn.quant.quant_layers.QuantizedLinear(
            layer=linear,
            weight_quantize_type='channel_wise_lsq_weight',
            activation_quantize_type='abs_max',
        )
        x = paddle.randn([4, 10], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [4, 5])


class TestQuantizedMatmul(unittest.TestCase):
    """测试 QuantizedMatmul 前向传播
    Test QuantizedMatmul forward pass"""

    def setUp(self):
        paddle.disable_static()

    def test_quantized_matmul_forward(self):
        """测试 QuantizedMatmul 前向传播 (line 1062-1125)
        Test QuantizedMatmul forward"""
        from paddle.nn.quant.quant_layers import QuantizedMatmul

        layer = QuantizedMatmul(
            activation_quantize_type='abs_max',
        )
        x = paddle.randn([2, 8], dtype='float32')
        y = paddle.randn([8, 4], dtype='float32')
        out = layer(x, y)
        self.assertEqual(out.shape, [2, 4])

    def test_quantized_matmul_with_act_quant_layer(self):
        """测试 QuantizedMatmul 带自定义激活量化层 (line 1084-1086)
        Test QuantizedMatmul with custom activation quant layer"""
        from paddle.nn.quant.quant_layers import QuantizedMatmul

        def custom_act_quant():
            return FakeQuantAbsMax(name='custom_matmul', quant_bits=8)

        layer = QuantizedMatmul(
            act_quant_layer=custom_act_quant,
        )
        x = paddle.randn([2, 8], dtype='float32')
        y = paddle.randn([8, 4], dtype='float32')
        out = layer(x, y)
        self.assertEqual(out.shape, [2, 4])

    def test_quantized_matmul_with_transpose(self):
        """测试 QuantizedMatmul 带转置 (line 1108-1125)
        Test QuantizedMatmul with transpose"""
        from paddle.nn.quant.quant_layers import QuantizedMatmul

        layer = QuantizedMatmul(
            activation_quantize_type='abs_max',
        )
        x = paddle.randn([8, 2], dtype='float32')
        y = paddle.randn([8, 4], dtype='float32')
        out = layer(x, y, transpose_x=True, transpose_y=False)
        self.assertEqual(out.shape, [2, 4])

    def test_quantized_matmul_with_pre_layers(self):
        """测试 QuantizedMatmul 带预处理层 (line 1101-1106, 1116-1121)
        Test QuantizedMatmul with preprocessing layers"""
        from paddle.nn.quant.quant_layers import QuantizedMatmul

        def act_pre():
            return paddle.nn.ReLU()

        layer = QuantizedMatmul(
            activation_quantize_type='abs_max',
            act_pre_layer=act_pre,
        )
        x = paddle.randn([2, 8], dtype='float32')
        y = paddle.randn([8, 4], dtype='float32')
        out = layer(x, y)
        self.assertEqual(out.shape, [2, 4])


class TestMAOutputScaleLayer(unittest.TestCase):
    """测试 MAOutputScaleLayer 行为
    Test MAOutputScaleLayer behavior"""

    def setUp(self):
        paddle.disable_static()

    def test_ma_output_scale_layer_forward(self):
        """测试 MAOutputScaleLayer 前向传播 (line 1128-1159)
        Test MAOutputScaleLayer forward"""
        base_layer = nn.Linear(4, 2)
        layer = MAOutputScaleLayer(layer=base_layer, moving_rate=0.9)
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 2])

    def test_ma_output_scale_with_dict_output(self):
        """测试 MAOutputScaleLayer 对 dict 输出的处理 (line 1156-1157)
        Test MAOutputScaleLayer with dict output (passthrough)"""

        # Use a layer that returns a dict-like output (list/tuple with >1 elements)
        # We create a custom layer that returns tuple
        class MultiOutLayer(nn.Layer):
            def forward(self, x):
                return (x, x * 2)

        base = MultiOutLayer()
        layer = MAOutputScaleLayer(layer=base, moving_rate=0.9)
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        # For list/tuple/dict output, returns directly without scaling
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)


class TestFakeQuantMAOutputScaleLayer(unittest.TestCase):
    """测试 FakeQuantMAOutputScaleLayer 行为
    Test FakeQuantMAOutputScaleLayer behavior"""

    def setUp(self):
        paddle.disable_static()

    def test_fake_quant_ma_output_scale_forward(self):
        """测试 FakeQuantMAOutputScaleLayer 前向传播 (line 1162-1196)
        Test FakeQuantMAOutputScaleLayer forward"""
        base_layer = nn.Linear(4, 2)
        layer = FakeQuantMAOutputScaleLayer(
            layer=base_layer, activation_bits=8, moving_rate=0.9
        )
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, [2, 2])

    def test_fake_quant_ma_output_scale_multi_output(self):
        """测试 FakeQuantMAOutputScaleLayer 对多输出的处理 (line 1193-1194)
        Test FakeQuantMAOutputScaleLayer with multi-output"""

        class MultiOutLayer(nn.Layer):
            def forward(self, x):
                return (x, x * 2, x * 3)

        base = MultiOutLayer()
        layer = FakeQuantMAOutputScaleLayer(
            layer=base, activation_bits=8, moving_rate=0.9
        )
        x = paddle.randn([2, 4], dtype='float32')
        out = layer(x)
        # len > 1 -> returns directly
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)


class TestGetFakeQuantType(unittest.TestCase):
    """测试 _get_fake_quant_type 工具函数的各条路径
    Test _get_fake_quant_type utility function paths"""

    def setUp(self):
        paddle.disable_static()

    def test_get_fake_quant_lsq_weight(self):
        """测试 _get_fake_quant_type 返回 lsq_weight 类型 (line 1219-1223)
        Test _get_fake_quant_type returns lsq_weight type"""
        from paddle.nn.quant.quant_layers import _get_fake_quant_type

        result = _get_fake_quant_type(
            'lsq_weight',
            quant_bits=8,
            quant_on_weight=True,
        )
        self.assertIsNotNone(result)

    def test_get_fake_quant_lsq_act(self):
        """测试 _get_fake_quant_type 返回 lsq_act 类型 (line 1234-1236)
        Test _get_fake_quant_type returns lsq_act type"""
        from paddle.nn.quant.quant_layers import _get_fake_quant_type

        result = _get_fake_quant_type(
            'lsq_act',
            quant_bits=8,
            symmetric=True,
        )
        self.assertIsNotNone(result)

    def test_get_fake_quant_channel_wise_lsq_weight(self):
        """测试 _get_fake_quant_type 返回 channel_wise_lsq_weight (line 1224-1233)
        Test _get_fake_quant_type returns channel_wise_lsq_weight"""
        from paddle.nn.quant.quant_layers import _get_fake_quant_type

        result = _get_fake_quant_type(
            'channel_wise_lsq_weight',
            quant_bits=8,
            channel_num=16,
            quant_on_weight=True,
        )
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
