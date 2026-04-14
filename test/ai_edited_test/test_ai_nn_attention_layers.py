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
Attention层单元测试 / Attention Layers Unit Tests

测试目标 / Test Target:
  paddle.nn Attention相关层

覆盖的模块 / Covered Modules:
  - paddle.nn.MultiHeadAttention: 多头注意力
  - paddle.nn.Transformer: Transformer完整架构
  - paddle.nn.TransformerEncoder/Decoder: 编码器/解码器

作用 / Purpose:
  补充注意力机制API的各种参数组合测试，提升覆盖率。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


class TestMultiHeadAttention(unittest.TestCase):
    """测试多头注意力 / Test multi-head attention"""

    def test_mha_basic(self):
        """测试基本多头注意力 / Test basic multi-head attention"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=8)
        q = paddle.randn([4, 10, 64])
        k = paddle.randn([4, 20, 64])
        v = paddle.randn([4, 20, 64])
        output = mha(q, k, v)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_mha_self_attention(self):
        """测试自注意力 / Test self-attention"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=8)
        x = paddle.randn([4, 10, 64])
        output = mha(x, x, x)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_mha_with_mask(self):
        """测试带掩码的多头注意力 / Test MHA with attention mask"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=8)
        q = paddle.randn([4, 10, 64])
        k = paddle.randn([4, 20, 64])
        v = paddle.randn([4, 20, 64])
        # Float attention mask (0 = attend, -inf = ignore)
        attn_mask = paddle.zeros([4, 1, 10, 20])
        output = mha(q, k, v, attn_mask=attn_mask)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_mha_dropout(self):
        """测试带dropout的多头注意力 / Test MHA with dropout"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=8, dropout=0.1)
        mha.train()
        x = paddle.randn([4, 10, 64])
        output = mha(x, x, x)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_mha_no_key_value(self):
        """测试只传query的MHA (自注意力) / Test MHA with only query (self-attention)"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=8)
        x = paddle.randn([4, 10, 64])
        output = mha(x)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_mha_different_key_value(self):
        """测试不同key和value的MHA / Test MHA with different key-value"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=8, kdim=32, vdim=16)
        q = paddle.randn([4, 10, 64])
        k = paddle.randn([4, 20, 32])
        v = paddle.randn([4, 20, 16])
        output = mha(q, k, v)
        self.assertEqual(output.shape, [4, 10, 64])


class TestTransformerEncoder(unittest.TestCase):
    """测试Transformer编码器 / Test Transformer encoder"""

    def test_encoder_layer(self):
        """测试TransformerEncoderLayer / Test TransformerEncoderLayer"""
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=256
        )
        src = paddle.randn([4, 10, 64])
        output = layer(src)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_encoder(self):
        """测试TransformerEncoder / Test TransformerEncoder"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=256
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        src = paddle.randn([4, 10, 64])
        output = encoder(src)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_encoder_layer_with_relu(self):
        """测试relu激活的Encoder层 / Test EncoderLayer with relu"""
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=256, activation='relu'
        )
        src = paddle.randn([4, 10, 64])
        output = layer(src)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_encoder_with_mask(self):
        """测试带掩码的Encoder / Test Encoder with mask"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=256
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        src = paddle.randn([4, 10, 64])
        # src_mask shape: [batch, n_head, seq, seq]
        src_mask = paddle.zeros([4, 8, 10, 10])
        output = encoder(src, src_mask=src_mask)
        self.assertEqual(output.shape, [4, 10, 64])


class TestTransformerDecoder(unittest.TestCase):
    """测试Transformer解码器 / Test Transformer decoder"""

    def test_decoder_layer(self):
        """测试TransformerDecoderLayer / Test TransformerDecoderLayer"""
        layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=8, dim_feedforward=256
        )
        tgt = paddle.randn([4, 8, 64])
        memory = paddle.randn([4, 10, 64])
        output = layer(tgt, memory)
        self.assertEqual(output.shape, [4, 8, 64])

    def test_decoder(self):
        """测试TransformerDecoder / Test TransformerDecoder"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=8, dim_feedforward=256
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        tgt = paddle.randn([4, 8, 64])
        memory = paddle.randn([4, 10, 64])
        output = decoder(tgt, memory)
        self.assertEqual(output.shape, [4, 8, 64])


class TestFullTransformer(unittest.TestCase):
    """测试完整Transformer / Test full Transformer"""

    def test_transformer_basic(self):
        """测试完整Transformer架构 / Test full Transformer architecture"""
        transformer = nn.Transformer(
            d_model=64,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
        )
        src = paddle.randn([4, 10, 64])
        tgt = paddle.randn([4, 8, 64])
        output = transformer(src, tgt)
        self.assertEqual(output.shape, [4, 8, 64])


if __name__ == '__main__':
    unittest.main()
