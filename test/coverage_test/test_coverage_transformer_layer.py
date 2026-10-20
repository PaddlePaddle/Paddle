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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.transformer
# 自动生成的单测，覆盖 paddle.nn.layer.transformer 模块中未覆盖的代码

"""
测试模块：paddle.nn.layer.transformer (TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, Transformer)
Test Module: paddle.nn.layer.transformer

本测试覆盖以下功能：
This test covers the following functions:
1. TransformerEncoderLayer - Transformer编码器层 / Transformer encoder layer with different activations
2. TransformerEncoder - Transformer编码器 / Transformer encoder with norm
3. TransformerDecoderLayer - Transformer解码器层 / Transformer decoder layer
4. Transformer - 完整Transformer / End-to-end Transformer

覆盖的未覆盖行：各层的不同activation分支, norm layer分支
"""

import unittest

import paddle


class TestTransformerEncoderLayer(unittest.TestCase):
    """测试TransformerEncoderLayer编码器层
    Test TransformerEncoderLayer"""

    def setUp(self):
        paddle.disable_static()

    def test_encoder_layer_relu(self):
        """ReLU激活 / Encoder layer with relu activation"""
        layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, activation='relu'
        )
        layer.eval()
        src = paddle.randn([2, 5, 64])
        out = layer(src)
        self.assertEqual(list(out.shape), [2, 5, 64])

    def test_encoder_layer_gelu(self):
        """GELU激活 / Encoder layer with gelu activation"""
        layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, activation='gelu'
        )
        layer.eval()
        src = paddle.randn([2, 5, 64])
        out = layer(src)
        self.assertEqual(list(out.shape), [2, 5, 64])

    def test_encoder_layer_with_mask(self):
        """带mask的编码器层 / Encoder layer with attention mask"""
        layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        layer.eval()
        src = paddle.randn([2, 5, 64])
        src_mask = paddle.zeros([5, 5])
        out = layer(src, src_mask=src_mask)
        self.assertEqual(list(out.shape), [2, 5, 64])


class TestTransformerEncoder(unittest.TestCase):
    """测试TransformerEncoder编码器
    Test TransformerEncoder"""

    def setUp(self):
        paddle.disable_static()

    def test_encoder_basic(self):
        """基本编码器 / Basic encoder"""
        encoder_layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        encoder = paddle.nn.TransformerEncoder(encoder_layer, num_layers=2)
        encoder.eval()
        src = paddle.randn([2, 5, 64])
        out = encoder(src)
        self.assertEqual(list(out.shape), [2, 5, 64])

    def test_encoder_with_norm(self):
        """带LayerNorm的编码器 / Encoder with final layer norm"""
        encoder_layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        norm = paddle.nn.LayerNorm(64)
        encoder = paddle.nn.TransformerEncoder(
            encoder_layer, num_layers=2, norm=norm
        )
        encoder.eval()
        src = paddle.randn([2, 5, 64])
        out = encoder(src)
        self.assertEqual(list(out.shape), [2, 5, 64])


class TestTransformerDecoderLayer(unittest.TestCase):
    """测试TransformerDecoderLayer解码器层
    Test TransformerDecoderLayer"""

    def setUp(self):
        paddle.disable_static()

    def test_decoder_layer_basic(self):
        """基本解码器层 / Basic decoder layer"""
        layer = paddle.nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        layer.eval()
        tgt = paddle.randn([2, 3, 64])
        memory = paddle.randn([2, 5, 64])
        out = layer(tgt, memory)
        self.assertEqual(list(out.shape), [2, 3, 64])

    def test_decoder_layer_with_mask(self):
        """带mask的解码器层 / Decoder layer with masks"""
        layer = paddle.nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        layer.eval()
        tgt = paddle.randn([2, 3, 64])
        memory = paddle.randn([2, 5, 64])
        tgt_mask = paddle.zeros([3, 3])
        out = layer(tgt, memory, tgt_mask=tgt_mask)
        self.assertEqual(list(out.shape), [2, 3, 64])


class TestTransformer(unittest.TestCase):
    """测试完整Transformer模型
    Test full Transformer model"""

    def setUp(self):
        paddle.disable_static()

    def test_transformer_basic(self):
        """基本Transformer / Basic Transformer"""
        transformer = paddle.nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
        )
        transformer.eval()
        src = paddle.randn([2, 5, 64])
        tgt = paddle.randn([2, 3, 64])
        out = transformer(src, tgt)
        self.assertEqual(list(out.shape), [2, 3, 64])


if __name__ == '__main__':
    unittest.main()
