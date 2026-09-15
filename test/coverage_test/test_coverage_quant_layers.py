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

# Unit test for paddle.nn.quant.quant_layers
# Target: cover QuantizedConv2D, QuantizedConv2DTranspose, QuantizedLinear

import unittest

import paddle
from paddle import nn
from paddle.nn.quant import quant_layers


class TestQuantizedConv2D(unittest.TestCase):
    """Test QuantizedConv2D layer.
    QuantizedConv2D takes an existing Conv2D layer as first argument.
    """

    def setUp(self):
        paddle.disable_static()

    def test_quantized_conv2d_basic(self):
        """Basic QuantizedConv2D initialization."""
        conv = nn.Conv2D(3, 16, 3, stride=1, padding=1)
        layer = quant_layers.QuantizedConv2D(layer=conv)
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_with_weight_quant(self):
        """QuantizedConv2D with weight quantize type."""
        conv = nn.Conv2D(3, 16, 3)
        layer = quant_layers.QuantizedConv2D(
            layer=conv,
            weight_quantize_type='channel_wise_abs_max',
        )
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_with_activation_quant(self):
        """QuantizedConv2D with activation quantize type."""
        conv = nn.Conv2D(3, 16, 3)
        layer = quant_layers.QuantizedConv2D(
            layer=conv,
            activation_quantize_type='moving_average_abs_max',
        )
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_with_groups(self):
        """QuantizedConv2D with groups."""
        conv = nn.Conv2D(4, 4, 3, groups=4)
        layer = quant_layers.QuantizedConv2D(layer=conv)
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_tuple_kernel(self):
        """QuantizedConv2D with tuple kernel_size."""
        conv = nn.Conv2D(3, 16, kernel_size=(3, 3))
        layer = quant_layers.QuantizedConv2D(layer=conv)
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_tuple_stride(self):
        """QuantizedConv2D with tuple stride."""
        conv = nn.Conv2D(3, 16, 3, stride=(2, 2))
        layer = quant_layers.QuantizedConv2D(layer=conv)
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_tuple_padding(self):
        """QuantizedConv2D with tuple padding."""
        conv = nn.Conv2D(3, 16, 3, padding=(1, 1))
        layer = quant_layers.QuantizedConv2D(layer=conv)
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_with_bias(self):
        """QuantizedConv2D with bias."""
        conv = nn.Conv2D(3, 16, 3, bias_attr=True)
        layer = quant_layers.QuantizedConv2D(layer=conv)
        self.assertIsNotNone(layer)


class TestQuantizedConv2DTranspose(unittest.TestCase):
    """Test QuantizedConv2DTranspose layer.
    QuantizedConv2DTranspose takes an existing Conv2DTranspose layer.
    """

    def setUp(self):
        paddle.disable_static()

    def test_quantized_conv2d_transpose_basic(self):
        """Basic QuantizedConv2DTranspose initialization."""
        conv = nn.Conv2DTranspose(
            16, 3, 3, stride=2, padding=1, output_padding=1
        )
        layer = quant_layers.QuantizedConv2DTranspose(layer=conv)
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_transpose_with_quant_types(self):
        """QuantizedConv2DTranspose with quantize types."""
        conv = nn.Conv2DTranspose(16, 3, 3)
        layer = quant_layers.QuantizedConv2DTranspose(
            layer=conv,
            weight_quantize_type='abs_max',
            activation_quantize_type='moving_average_abs_max',
        )
        self.assertIsNotNone(layer)

    def test_quantized_conv2d_transpose_tuple_params(self):
        """QuantizedConv2DTranspose with tuple params."""
        conv = nn.Conv2DTranspose(
            16, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        layer = quant_layers.QuantizedConv2DTranspose(layer=conv)
        self.assertIsNotNone(layer)


class TestQuantizedLinear(unittest.TestCase):
    """Test QuantizedLinear layer.
    QuantizedLinear takes an existing Linear layer.
    """

    def setUp(self):
        paddle.disable_static()

    def test_quantized_linear_basic(self):
        """Basic QuantizedLinear initialization."""
        linear = nn.Linear(10, 5)
        layer = quant_layers.QuantizedLinear(layer=linear)
        self.assertIsNotNone(layer)

    def test_quantized_linear_with_quant_types(self):
        """QuantizedLinear with quantize types."""
        linear = nn.Linear(10, 5)
        layer = quant_layers.QuantizedLinear(
            layer=linear,
            weight_quantize_type='channel_wise_abs_max',
            activation_quantize_type='moving_average_abs_max',
        )
        self.assertIsNotNone(layer)

    def test_quantized_linear_with_bias(self):
        """QuantizedLinear with bias."""
        linear = nn.Linear(10, 5, bias_attr=True)
        layer = quant_layers.QuantizedLinear(layer=linear)
        self.assertIsNotNone(layer)


if __name__ == '__main__':
    unittest.main()
