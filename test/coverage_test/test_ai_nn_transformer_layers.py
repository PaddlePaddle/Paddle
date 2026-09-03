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

# [AUTO-GENERATED] Test file for paddle.nn.layer.transformer
# 覆盖模块: paddle/nn/layer/transformer.py
# 未覆盖行: 73,76,77,78,79,80,82,84,87,88,89,91,98,126,293,299,300,301,399,400,401,402,403,404,405,406,407,410,530,562
# Covered module: paddle/nn/layer/transformer.py
# Uncovered lines: 73,76-80,82,84,87-89,91,98,126,293,299-301,399-407,410,530,562

import unittest

import paddle


class TestMultiHeadAttention(unittest.TestCase):
    """测试 MultiHeadAttention 类
    Test MultiHeadAttention class"""

    def test_mha_init(self):
        """测试 MultiHeadAttention 初始化
        Test MultiHeadAttention initialization"""
        mha = paddle.nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        self.assertIsNotNone(mha)

    def test_mha_forward(self):
        """测试 MultiHeadAttention 前向传播
        Test MultiHeadAttention forward pass"""
        mha = paddle.nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        query = paddle.randn([2, 5, 64])
        key = paddle.randn([2, 5, 64])
        value = paddle.randn([2, 5, 64])
        output = mha(query, key, value)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_mha_with_attn_mask(self):
        """测试带注意力掩码的 MultiHeadAttention
        Test MultiHeadAttention with attention mask"""
        mha = paddle.nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        query = paddle.randn([2, 5, 64])
        key = paddle.randn([2, 5, 64])
        value = paddle.randn([2, 5, 64])
        attn_mask = paddle.randn([2, 4, 5, 5])
        output = mha(query, key, value, attn_mask=attn_mask)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_mha_with_kdim_vdim(self):
        """测试带不同 kdim/vdim 的 MultiHeadAttention
        Test MultiHeadAttention with different kdim/vdim"""
        mha = paddle.nn.MultiHeadAttention(
            embed_dim=64, num_heads=4, kdim=32, vdim=32
        )
        query = paddle.randn([2, 5, 64])
        key = paddle.randn([2, 5, 32])
        value = paddle.randn([2, 5, 32])
        output = mha(query, key, value)
        self.assertEqual(output.shape, [2, 5, 64])


class TestTransformerEncoderLayer(unittest.TestCase):
    """测试 TransformerEncoderLayer 类
    Test TransformerEncoderLayer class"""

    def test_encoder_layer_init(self):
        """测试 TransformerEncoderLayer 初始化
        Test TransformerEncoderLayer initialization"""
        layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        self.assertIsNotNone(layer)

    def test_encoder_layer_forward(self):
        """测试 TransformerEncoderLayer 前向传播
        Test TransformerEncoderLayer forward pass"""
        layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        src = paddle.randn([2, 5, 64])
        output = layer(src)
        self.assertEqual(output.shape, [2, 5, 64])


class TestTransformerDecoderLayer(unittest.TestCase):
    """测试 TransformerDecoderLayer 类
    Test TransformerDecoderLayer class"""

    def test_decoder_layer_init(self):
        """测试 TransformerDecoderLayer 初始化
        Test TransformerDecoderLayer initialization"""
        layer = paddle.nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        self.assertIsNotNone(layer)

    def test_decoder_layer_forward(self):
        """测试 TransformerDecoderLayer 前向传播
        Test TransformerDecoderLayer forward pass"""
        layer = paddle.nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        tgt = paddle.randn([2, 5, 64])
        memory = paddle.randn([2, 10, 64])
        output = layer(tgt, memory)
        self.assertEqual(output.shape, [2, 5, 64])


class TestTransformerEncoder(unittest.TestCase):
    """测试 TransformerEncoder 类
    Test TransformerEncoder class"""

    def test_encoder_init(self):
        """测试 TransformerEncoder 初始化
        Test TransformerEncoder initialization"""
        encoder_layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        encoder = paddle.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.assertIsNotNone(encoder)

    def test_encoder_forward(self):
        """测试 TransformerEncoder 前向传播
        Test TransformerEncoder forward pass"""
        encoder_layer = paddle.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        encoder = paddle.nn.TransformerEncoder(encoder_layer, num_layers=2)
        src = paddle.randn([2, 5, 64])
        output = encoder(src)
        self.assertEqual(output.shape, [2, 5, 64])


class TestTransformerDecoder(unittest.TestCase):
    """测试 TransformerDecoder 类
    Test TransformerDecoder class"""

    def test_decoder_init(self):
        """测试 TransformerDecoder 初始化
        Test TransformerDecoder initialization"""
        decoder_layer = paddle.nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        decoder = paddle.nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.assertIsNotNone(decoder)

    def test_decoder_forward(self):
        """测试 TransformerDecoder 前向传播
        Test TransformerDecoder forward pass"""
        decoder_layer = paddle.nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        decoder = paddle.nn.TransformerDecoder(decoder_layer, num_layers=2)
        tgt = paddle.randn([2, 5, 64])
        memory = paddle.randn([2, 10, 64])
        output = decoder(tgt, memory)
        self.assertEqual(output.shape, [2, 5, 64])


if __name__ == '__main__':
    unittest.main()
