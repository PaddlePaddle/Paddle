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
# Target: paddle/nn/layer/transformer.py

"""
测试模块：paddle.nn.layer.transformer
Test Module: paddle.nn.layer.transformer

本测试覆盖以下功能：
This test covers the following functions:
1. MultiHeadAttention - 多头注意力机制 / Multi-head attention with different kdim/vdim, need_weights, cache
2. TransformerEncoderLayer - 编码器层 / Encoder layer with normalize_before, cache, gen_cache
3. TransformerDecoderLayer - 解码器层 / Decoder layer with cache, cross attention, normalize_before
4. TransformerEncoder - 编码器 / Encoder with norm, gen_cache
5. TransformerDecoder - 解码器 / Decoder with norm, gen_cache, do_zip
6. Transformer - 完整Transformer / Full transformer with custom encoder/decoder, bias_attr list
7. _convert_param_attr_to_list - 参数属性转换 / param attr conversion helper
8. _convert_attention_mask - 注意力掩码转换 / attention mask conversion
"""

import unittest

import paddle
from paddle import nn


class TestMultiHeadAttentionComprehensive(unittest.TestCase):
    """测试MultiHeadAttention多头注意力
    Test MultiHeadAttention comprehensive scenarios"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_mha_with_kdim_vdim(self):
        """测试不同的kdim和vdim / Test with different kdim and vdim"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4, kdim=32, vdim=48)
        self.assertEqual(mha.kdim, 32)
        self.assertEqual(mha.vdim, 48)
        query = paddle.randn([2, 5, 64])
        key = paddle.randn([2, 5, 32])
        value = paddle.randn([2, 5, 48])
        output = mha(query, key, value)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_mha_with_need_weights(self):
        """测试返回注意力权重 / Test with need_weights=True"""
        mha = nn.MultiHeadAttention(
            embed_dim=64, num_heads=4, need_weights=True
        )
        query = paddle.randn([2, 5, 64])
        attn_mask = paddle.randn([2, 4, 5, 5])
        output, weights = mha(query, query, query, attn_mask=attn_mask)
        self.assertEqual(output.shape, [2, 5, 64])
        self.assertEqual(weights.shape, [2, 4, 5, 5])

    def test_mha_with_dropout(self):
        """测试带dropout的多头注意力 / Test MHA with dropout in training mode"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.1)
        mha.train()
        query = paddle.randn([2, 5, 64])
        output = mha(query)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_mha_gen_cache_static(self):
        """测试生成StaticCache / Test gen_cache with StaticCache type"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        key = paddle.randn([2, 5, 64])
        value = paddle.randn([2, 5, 64])
        cache = mha.gen_cache(key, value, type=mha.StaticCache)
        self.assertIsInstance(cache, mha.StaticCache)
        self.assertEqual(cache.k.shape, [2, 4, 5, 16])
        self.assertEqual(cache.v.shape, [2, 4, 5, 16])

    def test_mha_gen_cache_with_value(self):
        """测试生成Cache并传入value / Test gen_cache with Cache type and value"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        key = paddle.randn([2, 5, 64])
        cache = mha.gen_cache(key, value=None, type=mha.Cache)
        self.assertIsInstance(cache, mha.Cache)
        self.assertEqual(cache.k.shape[2], 0)  # empty for incremental
        self.assertEqual(cache.v.shape[2], 0)

    def test_mha_gen_cache_with_initial_value(self):
        """测试生成Cache并传入初始value / Test gen_cache with initial value for UniLM-like usage"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        key = paddle.randn([2, 5, 64])
        value = paddle.randn([2, 5, 64])
        cache = mha.gen_cache(key, value=value, type=mha.Cache)
        self.assertIsInstance(cache, mha.Cache)
        self.assertEqual(cache.k.shape, [2, 5, 64])
        self.assertEqual(cache.v.shape, [2, 5, 64])

    def test_mha_with_cache_forward(self):
        """测试带cache的forward / Test forward with Cache"""
        mha = nn.MultiHeadAttention(
            embed_dim=64, num_heads=4, need_weights=True
        )
        mha.eval()
        query = paddle.randn([2, 3, 64])
        key = paddle.randn([2, 5, 64])
        value = paddle.randn([2, 5, 64])
        cache = mha.gen_cache(key, value, type=mha.StaticCache)
        output, weights, new_cache = mha(query, key, value, cache=cache)
        self.assertEqual(output.shape, [2, 3, 64])
        self.assertEqual(weights.shape, [2, 4, 3, 5])
        self.assertEqual(new_cache.k.shape, [2, 4, 5, 16])

    def test_mha_compute_kv(self):
        """测试compute_kv方法 / Test compute_kv method"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        key = paddle.randn([2, 5, 64])
        value = paddle.randn([2, 5, 64])
        k, v = mha.compute_kv(key, value)
        self.assertEqual(k.shape, [2, 4, 5, 16])
        self.assertEqual(v.shape, [2, 4, 5, 16])

    def test_mha_with_bool_mask(self):
        """测试bool类型注意力掩码 / Test with bool attention mask"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        query = paddle.randn([2, 5, 64])
        attn_mask = paddle.ones([2, 4, 5, 5], dtype='bool')
        output = mha(query, query, query, attn_mask=attn_mask)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_mha_with_int_mask(self):
        """测试int类型注意力掩码 / Test with int attention mask"""
        mha = nn.MultiHeadAttention(embed_dim=64, num_heads=4)
        query = paddle.randn([2, 5, 64])
        attn_mask = paddle.ones([2, 4, 5, 5], dtype='int64')
        output = mha(query, query, query, attn_mask=attn_mask)
        self.assertEqual(output.shape, [2, 5, 64])


class TestTransformerEncoderLayerComprehensive(unittest.TestCase):
    """测试TransformerEncoderLayer
    Test TransformerEncoderLayer"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_encoder_layer_normalize_before(self):
        """测试pre-norm编码器层 / Test encoder layer with normalize_before=True"""
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            activation='relu',
            normalize_before=True,
        )
        layer.eval()
        src = paddle.randn([2, 5, 64])
        output = layer(src)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_encoder_layer_gelu_activation(self):
        """测试gelu激活函数 / Test encoder layer with gelu activation"""
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, activation='gelu'
        )
        layer.eval()
        src = paddle.randn([2, 5, 64])
        output = layer(src)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_encoder_layer_with_mask(self):
        """测试带掩码的编码器层 / Test encoder layer with mask"""
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        layer.eval()
        src = paddle.randn([2, 5, 64])
        src_mask = paddle.randn([2, 4, 5, 5])
        output = layer(src, src_mask=src_mask)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_encoder_layer_gen_cache(self):
        """测试编码器层gen_cache / Test encoder layer gen_cache"""
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        src = paddle.randn([2, 5, 64])
        cache = layer.gen_cache(src)
        self.assertIsInstance(cache, nn.MultiHeadAttention.Cache)

    def test_encoder_layer_with_cache_forward(self):
        """测试带cache的编码器层forward / Test encoder layer forward with cache"""
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        layer.eval()
        src = paddle.randn([2, 3, 64])
        cache = layer.gen_cache(src)
        output, new_cache = layer(src, cache=cache)
        self.assertEqual(output.shape, [2, 3, 64])
        self.assertIsInstance(new_cache, nn.MultiHeadAttention.Cache)

    def test_encoder_layer_with_bias_attr_list(self):
        """测试列表形式的bias_attr / Test with list bias_attr"""
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, bias_attr=[False, False]
        )
        layer.eval()
        src = paddle.randn([2, 5, 64])
        output = layer(src)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_encoder_layer_attn_act_dropout(self):
        """测试独立的attn_dropout和act_dropout / Test separate attn and act dropout"""
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            attn_dropout=0.2,
            act_dropout=0.3,
        )
        layer.eval()
        src = paddle.randn([2, 5, 64])
        output = layer(src)
        self.assertEqual(output.shape, [2, 5, 64])


class TestTransformerEncoderComprehensive(unittest.TestCase):
    """测试TransformerEncoder
    Test TransformerEncoder"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_encoder_with_norm(self):
        """测试带norm的编码器 / Test encoder with norm layer"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        norm = nn.LayerNorm(64)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=norm)
        encoder.eval()
        src = paddle.randn([2, 5, 64])
        output = encoder(src)
        self.assertEqual(output.shape, [2, 5, 64])

    def test_encoder_gen_cache(self):
        """测试编码器gen_cache / Test encoder gen_cache"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        src = paddle.randn([2, 5, 64])
        cache_list = encoder.gen_cache(src)
        self.assertEqual(len(cache_list), 2)
        for cache in cache_list:
            self.assertIsInstance(cache, nn.MultiHeadAttention.Cache)

    def test_encoder_with_cache(self):
        """测试带cache的编码器 / Test encoder with cache"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        encoder.eval()
        src = paddle.randn([2, 3, 64])
        cache = encoder.gen_cache(src)
        output, new_caches = encoder(src, cache=cache)
        self.assertEqual(output.shape, [2, 3, 64])
        self.assertEqual(len(new_caches), 2)


class TestTransformerDecoderLayerComprehensive(unittest.TestCase):
    """测试TransformerDecoderLayer
    Test TransformerDecoderLayer"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_decoder_layer_normalize_before(self):
        """测试pre-norm解码器层 / Test decoder layer with normalize_before=True"""
        layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, normalize_before=True
        )
        layer.eval()
        tgt = paddle.randn([2, 4, 64])
        memory = paddle.randn([2, 6, 64])
        output = layer(tgt, memory)
        self.assertEqual(output.shape, [2, 4, 64])

    def test_decoder_layer_with_masks(self):
        """测试带掩码的解码器层 / Test decoder layer with tgt_mask and memory_mask"""
        layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        layer.eval()
        tgt = paddle.randn([2, 4, 64])
        memory = paddle.randn([2, 6, 64])
        tgt_mask = paddle.randn([2, 4, 4, 4])
        memory_mask = paddle.randn([2, 4, 4, 6])
        output = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        self.assertEqual(output.shape, [2, 4, 64])

    def test_decoder_layer_gen_cache(self):
        """测试解码器层gen_cache / Test decoder layer gen_cache"""
        layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        memory = paddle.randn([2, 6, 64])
        incremental_cache, static_cache = layer.gen_cache(memory)
        self.assertIsInstance(incremental_cache, nn.MultiHeadAttention.Cache)
        self.assertIsInstance(static_cache, nn.MultiHeadAttention.StaticCache)

    def test_decoder_layer_with_cache(self):
        """测试带cache的解码器层forward / Test decoder layer with cache"""
        layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        layer.eval()
        tgt = paddle.randn([2, 3, 64])
        memory = paddle.randn([2, 6, 64])
        cache = layer.gen_cache(memory)
        output, new_cache = layer(tgt, memory, cache=cache)
        self.assertEqual(output.shape, [2, 3, 64])
        self.assertIsInstance(new_cache[0], nn.MultiHeadAttention.Cache)
        self.assertIsInstance(new_cache[1], nn.MultiHeadAttention.StaticCache)

    def test_decoder_layer_gelu(self):
        """测试gelu激活函数的解码器层 / Test decoder layer with gelu"""
        layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, activation='gelu'
        )
        layer.eval()
        tgt = paddle.randn([2, 4, 64])
        memory = paddle.randn([2, 6, 64])
        output = layer(tgt, memory)
        self.assertEqual(output.shape, [2, 4, 64])

    def test_decoder_layer_with_bias_attr_list(self):
        """测试列表形式的bias_attr / Test with list bias_attr"""
        layer = nn.TransformerDecoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            bias_attr=[False, False, False],
        )
        layer.eval()
        tgt = paddle.randn([2, 4, 64])
        memory = paddle.randn([2, 6, 64])
        output = layer(tgt, memory)
        self.assertEqual(output.shape, [2, 4, 64])


class TestTransformerDecoderComprehensive(unittest.TestCase):
    """测试TransformerDecoder
    Test TransformerDecoder"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_decoder_with_norm(self):
        """测试带norm的解码器 / Test decoder with norm"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        norm = nn.LayerNorm(64)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=2, norm=norm)
        decoder.eval()
        tgt = paddle.randn([2, 4, 64])
        memory = paddle.randn([2, 6, 64])
        output = decoder(tgt, memory)
        self.assertEqual(output.shape, [2, 4, 64])

    def test_decoder_gen_cache(self):
        """测试解码器gen_cache / Test decoder gen_cache"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        memory = paddle.randn([2, 6, 64])
        cache = decoder.gen_cache(memory)
        self.assertEqual(len(cache), 2)
        for inc, sc in cache:
            self.assertIsInstance(inc, nn.MultiHeadAttention.Cache)
            self.assertIsInstance(sc, nn.MultiHeadAttention.StaticCache)

    def test_decoder_gen_cache_do_zip(self):
        """测试gen_cache with do_zip / Test gen_cache with do_zip=True"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        memory = paddle.randn([2, 6, 64])
        cache = decoder.gen_cache(memory, do_zip=True)
        self.assertEqual(len(cache), 2)

    def test_decoder_with_cache(self):
        """测试带cache的解码器 / Test decoder with cache"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=4, dim_feedforward=128
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        decoder.eval()
        tgt = paddle.randn([2, 3, 64])
        memory = paddle.randn([2, 6, 64])
        cache = decoder.gen_cache(memory)
        output, new_caches = decoder(tgt, memory, cache=cache)
        self.assertEqual(output.shape, [2, 3, 64])
        self.assertEqual(len(new_caches), 2)


class TestTransformerComprehensive(unittest.TestCase):
    """测试Transformer完整模型
    Test Transformer full model"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_transformer_basic(self):
        """测试基本Transformer / Test basic Transformer"""
        transformer = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
        )
        transformer.eval()
        src = paddle.randn([2, 5, 64])
        tgt = paddle.randn([2, 3, 64])
        output = transformer(src, tgt)
        self.assertEqual(output.shape, [2, 3, 64])

    def test_transformer_with_masks(self):
        """测试带掩码的Transformer / Test Transformer with masks"""
        transformer = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
        )
        transformer.eval()
        src = paddle.randn([2, 5, 64])
        tgt = paddle.randn([2, 3, 64])
        src_mask = paddle.randn([2, 4, 5, 5])
        tgt_mask = paddle.randn([2, 4, 3, 3])
        memory_mask = paddle.randn([2, 4, 3, 5])
        output = transformer(src, tgt, src_mask, tgt_mask, memory_mask)
        self.assertEqual(output.shape, [2, 3, 64])

    def test_transformer_generate_square_subsequent_mask(self):
        """测试生成方阵因果掩码 / Test generate_square_subsequent_mask"""
        transformer = nn.Transformer(d_model=64, nhead=4, dim_feedforward=128)
        mask = transformer.generate_square_subsequent_mask(5)
        self.assertEqual(mask.shape, [5, 5])
        self.assertTrue(paddle.all(mask[0, 0] == 0.0))
        self.assertTrue(paddle.isinf(mask[0, 1]))

    def test_transformer_with_bias_attr_list(self):
        """测试列表形式的bias_attr / Test with list bias_attr"""
        transformer = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
            bias_attr=[False],
        )
        transformer.eval()
        src = paddle.randn([2, 5, 64])
        tgt = paddle.randn([2, 3, 64])
        output = transformer(src, tgt)
        self.assertEqual(output.shape, [2, 3, 64])

    def test_transformer_with_weight_attr_list(self):
        """测试列表形式的weight_attr / Test with list weight_attr"""
        transformer = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
            weight_attr=[paddle.nn.initializer.XavierNormal()],
        )
        transformer.eval()
        src = paddle.randn([2, 5, 64])
        tgt = paddle.randn([2, 3, 64])
        output = transformer(src, tgt)
        self.assertEqual(output.shape, [2, 3, 64])


class TestConvertAttentionMask(unittest.TestCase):
    """测试_convert_attention_mask辅助函数
    Test _convert_attention_mask helper function"""

    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_convert_mask_none(self):
        """测试None掩码 / Test with None mask"""
        from paddle.nn.layer.transformer import _convert_attention_mask

        result = _convert_attention_mask(None, 'float32')
        self.assertIsNone(result)

    def test_convert_mask_same_dtype(self):
        """测试相同数据类型掩码 / Test mask with same dtype"""
        from paddle.nn.layer.transformer import _convert_attention_mask

        mask = paddle.randn([2, 4, 5, 5])
        result = _convert_attention_mask(mask, mask.dtype)
        self.assertIs(result, mask)

    def test_convert_mask_bool_to_float(self):
        """测试bool掩码转float / Test bool mask to float conversion"""
        from paddle.nn.layer.transformer import _convert_attention_mask

        mask = paddle.ones([2, 4, 5, 5], dtype='bool')
        result = _convert_attention_mask(mask, 'float32')
        self.assertEqual(result.dtype, paddle.float32)


class TestConvertParamAttrToList(unittest.TestCase):
    """测试_convert_param_attr_to_list辅助函数
    Test _convert_param_attr_to_list helper function"""

    def test_convert_none_param(self):
        """测试None参数 / Test with None param_attr"""
        from paddle.nn.layer.transformer import _convert_param_attr_to_list

        result = _convert_param_attr_to_list(None, 3)
        self.assertEqual(len(result), 3)

    def test_convert_bool_true_param(self):
        """测试bool True参数 / Test with True param_attr"""
        from paddle.nn.layer.transformer import _convert_param_attr_to_list

        result = _convert_param_attr_to_list(True, 2)
        self.assertEqual(len(result), 2)

    def test_convert_bool_false_param(self):
        """测试bool False参数 / Test with False param_attr"""
        from paddle.nn.layer.transformer import _convert_param_attr_to_list

        result = _convert_param_attr_to_list(False, 2)
        self.assertEqual(result, [False, False])

    def test_convert_list_param(self):
        """测试列表参数 / Test with list param_attr"""
        from paddle.nn.layer.transformer import _convert_param_attr_to_list

        result = _convert_param_attr_to_list([False, True], 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], False)


class TestMultiHeadAssertionErrors(unittest.TestCase):
    """测试MultiHeadAttention断言错误
    Test MultiHeadAttention assertion errors"""

    def test_mha_zero_embed_dim(self):
        """测试embed_dim为0时断言 / Test assertion with embed_dim=0"""
        with self.assertRaises(AssertionError):
            nn.MultiHeadAttention(embed_dim=0, num_heads=4)

    def test_mha_zero_num_heads(self):
        """测试num_heads为0时断言 / Test assertion with num_heads=0"""
        with self.assertRaises(AssertionError):
            nn.MultiHeadAttention(embed_dim=64, num_heads=0)

    def test_mha_indivisible(self):
        """测试embed_dim不能被num_heads整除时断言 / Test when embed_dim not divisible by num_heads"""
        with self.assertRaises(AssertionError):
            nn.MultiHeadAttention(embed_dim=64, num_heads=3)

    def test_encoder_layer_zero_d_model(self):
        """测试d_model为0时断言 / Test assertion with d_model=0"""
        with self.assertRaises(AssertionError):
            nn.TransformerEncoderLayer(d_model=0, nhead=4, dim_feedforward=128)


if __name__ == '__main__':
    unittest.main()
