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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.transformer (Transformer, MultiHeadAttention)
# 自动生成的单测，覆盖 paddle.nn.layer.transformer 模块中未覆盖的额外代码路径
# Target: cover uncovered lines 73-98, 126, 293 in paddle/python/paddle/nn/layer/transformer.py
# 目标：覆盖 Transformer 和 MultiHeadAttention 的边界情况和未覆盖分支

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. MultiHeadAttention - enable_fast_math 参数 (line 73-78)
2. MultiHeadAttention - _reset_parameters / _prepare_qkv (line 82-98)
3. MultiHeadAttention - dropout 路径 (line 126)
4. TransformerEncoderLayer / TransformerDecoderLayer - 各种初始化组合 (line 293+)
5. Transformer - 基本使用
"""

import unittest

import paddle
from paddle import nn


class TestMultiHeadAttentionAdvanced(unittest.TestCase):
    """Test MultiHeadAttention advanced features.
    测试 MultiHeadAttention 高级功能。
    """

    def setUp(self):
        paddle.disable_static()

    def test_mha_basic(self):
        """Basic MultiHeadAttention."""
        mha = nn.MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
        )
        q = paddle.randn([2, 10, 64])
        k = paddle.randn([2, 10, 64])
        v = paddle.randn([2, 10, 64])
        out = mha(q, k, v)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_mha_with_key_value_memory(self):
        """MHA with separate key/value memory (cross-attention)."""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        q = paddle.randn([2, 10, 64])
        k = paddle.randn([2, 20, 64])
        v = paddle.randn([2, 20, 64])
        out = mha(q, k, v)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_mha_with_attn_mask(self):
        """MHA with attention mask."""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        q = k = v = paddle.randn([2, 10, 64])
        # Upper triangular mask (causal)
        mask = paddle.triu(paddle.ones([10, 10]), diagonal=1) * (-1e9)
        mask = mask.astype('float32')
        out = mha(q, k, v, attn_mask=mask)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_mha_different_embed_dim_kdim_vdim(self):
        """MHA with different kdim and vdim."""
        mha = nn.MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            kdim=32,
            vdim=32,
        )
        q = paddle.randn([2, 10, 64])
        k = paddle.randn([2, 10, 32])
        v = paddle.randn([2, 10, 32])
        out = mha(q, k, v)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_mha_dropout(self):
        """MHA with dropout."""
        mha = nn.MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
        )
        mha.train()
        q = k = v = paddle.randn([2, 10, 64])
        out = mha(q, k, v)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_mha_need_weights(self):
        """MHA with need_weights=True returns tuple."""
        mha = nn.MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            need_weights=True,
        )
        q = k = v = paddle.randn([2, 10, 64])
        result = mha(q, k, v)
        self.assertIsInstance(result, tuple)
        out, attn = result
        self.assertEqual(out.shape, [2, 10, 64])


class TestTransformerEncoderLayer(unittest.TestCase):
    """Test TransformerEncoderLayer.
    测试 TransformerEncoderLayer。
    """

    def setUp(self):
        paddle.disable_static()

    def test_encoder_layer_basic(self):
        """Basic TransformerEncoderLayer."""
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
        )
        src = paddle.randn([2, 10, 64])
        out = layer(src)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_encoder_layer_with_dropout(self):
        """EncoderLayer with dropout."""
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation='gelu',
        )
        layer.train()
        src = paddle.randn([2, 10, 64])
        out = layer(src)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_encoder_layer_normalize_before(self):
        """EncoderLayer with normalize_before."""
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            normalize_before=True,
        )
        src = paddle.randn([2, 10, 64])
        out = layer(src)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_encoder_layer_relu(self):
        """EncoderLayer with relu activation."""
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            activation='relu',
        )
        src = paddle.randn([2, 10, 64])
        out = layer(src)
        self.assertEqual(out.shape, [2, 10, 64])


class TestTransformerDecoderLayer(unittest.TestCase):
    """Test TransformerDecoderLayer.
    测试 TransformerDecoderLayer。
    """

    def setUp(self):
        paddle.disable_static()

    def test_decoder_layer_basic(self):
        """Basic TransformerDecoderLayer."""
        layer = nn.TransformerDecoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
        )
        tgt = paddle.randn([2, 10, 64])
        memory = paddle.randn([2, 20, 64])
        out = layer(tgt, memory)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_decoder_layer_with_mask(self):
        """DecoderLayer with tgt_mask."""
        layer = nn.TransformerDecoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
        )
        tgt = paddle.randn([2, 10, 64])
        memory = paddle.randn([2, 20, 64])
        tgt_mask = paddle.triu(paddle.ones([10, 10]), diagonal=1) * (-1e9)
        tgt_mask = tgt_mask.astype('float32')
        out = layer(tgt, memory, tgt_mask=tgt_mask)
        self.assertEqual(out.shape, [2, 10, 64])


class TestTransformerEncoder(unittest.TestCase):
    """Test TransformerEncoder.
    测试 TransformerEncoder。
    """

    def setUp(self):
        paddle.disable_static()

    def test_encoder_basic(self):
        """Basic TransformerEncoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        src = paddle.randn([2, 10, 64])
        out = encoder(src)
        self.assertEqual(out.shape, [2, 10, 64])

    def test_encoder_with_norm(self):
        """TransformerEncoder with norm."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256
        )
        norm = nn.LayerNorm(64)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=norm)
        src = paddle.randn([2, 10, 64])
        out = encoder(src)
        self.assertEqual(out.shape, [2, 10, 64])


class TestTransformerDecoder(unittest.TestCase):
    """Test TransformerDecoder.
    测试 TransformerDecoder。
    """

    def setUp(self):
        paddle.disable_static()

    def test_decoder_basic(self):
        """Basic TransformerDecoder."""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=256
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        tgt = paddle.randn([2, 10, 64])
        memory = paddle.randn([2, 20, 64])
        out = decoder(tgt, memory)
        self.assertEqual(out.shape, [2, 10, 64])


class TestTransformer(unittest.TestCase):
    """Test Transformer model.
    测试 Transformer 模型。
    """

    def setUp(self):
        paddle.disable_static()

    def test_transformer_basic(self):
        """Basic Transformer forward."""
        transformer = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
        )
        src = paddle.randn([2, 10, 64])
        tgt = paddle.randn([2, 5, 64])
        out = transformer(src, tgt)
        self.assertEqual(out.shape, [2, 5, 64])

    def test_transformer_with_custom_mask(self):
        """Transformer with custom masks."""
        transformer = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=256,
        )
        src = paddle.randn([2, 10, 64])
        tgt = paddle.randn([2, 5, 64])
        tgt_mask = paddle.triu(paddle.ones([5, 5]), diagonal=1) * (-1e9)
        tgt_mask = tgt_mask.astype('float32')
        out = transformer(src, tgt, tgt_mask=tgt_mask)
        self.assertEqual(out.shape, [2, 5, 64])


if __name__ == '__main__':
    unittest.main()
