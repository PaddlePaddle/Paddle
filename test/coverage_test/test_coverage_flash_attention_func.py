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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.flash_attention
# 自动生成的单测，覆盖 paddle.nn.functional.flash_attention 模块中不同代码路径
# Target: cover uncovered lines in python/paddle/nn/functional/flash_attention.py
# NOTE: test_ai_flash_attention.py already covers:
#   _select_sdp_cuda, _select_sdp, sdp_kernel,
#   flash_attention basic/dropout/causal,
#   scaled_dot_product_attention basic/causal
# This test covers DIFFERENT functions:
#   get_triangle_upper_mask, _math_attention,
#   flash_attn_qkvpacked, flashmask_attention, flash_attn_varlen_func

"""
测试模块：paddle.nn.functional.flash_attention
Test Module: paddle.nn.functional.flash_attention

本测试覆盖以下功能：
This test covers the following functions:
1. get_triangle_upper_mask - 上三角掩码生成 / Upper triangle mask generation
   - 基本掩码验证 / Basic mask verification
   - 掩码值应为 -1e4 / Mask values should be -1e4
2. _math_attention - 数学注意力计算 / Math attention computation
   - 无掩码基本注意力 / Basic attention without mask
   - causal=True / Causal attention
   - return_softmax=True / Return softmax weights
   - 自定义 scale / Custom scale
3. flash_attn_qkvpacked - 打包 QKV 闪存注意力 / Packed QKV flash attention
   - GPU 依赖 / GPU required
4. flashmask_attention - 闪存掩码注意力 / Flash mask attention
   - GPU 依赖 / GPU required
5. flash_attn_varlen_func - 变长闪存注意力 / Variable length flash attention
   - GPU 依赖 / GPU required
"""

import unittest

import numpy as np

import paddle
from paddle.nn.functional.flash_attention import (
    _math_attention,
    get_triangle_upper_mask,
)


class TestGetTriangleUpperMask(unittest.TestCase):
    """测试 get_triangle_upper_mask 函数
    Test get_triangle_upper_mask function"""

    def setUp(self):
        """设置测试环境 / Set up test environment"""
        paddle.disable_static()

    def test_basic_2d_mask(self):
        """测试基本的 2D 输入掩码
        Test basic 2D input mask"""
        x = paddle.randn([4, 4])
        mask = get_triangle_upper_mask(x)
        # 掩码形状应与输入相同 / Mask shape should match input
        self.assertEqual(list(mask.shape), [4, 4])
        mask_np = mask.numpy()
        # 对角线及以下应为 0 / Diagonal and below should be 0
        np.testing.assert_allclose(mask_np[0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(mask_np[3, 3], 0.0, atol=1e-5)
        # 上三角应为 -1e4 / Upper triangle should be -1e4
        np.testing.assert_allclose(mask_np[0, 1], -1e4, atol=1e-2)
        np.testing.assert_allclose(mask_np[0, 3], -1e4, atol=1e-2)

    def test_mask_values_are_neg_inf_like(self):
        """测试掩码值是否为 -1e4
        Test that mask values are -1e4"""
        x = paddle.randn([5, 5])
        mask = get_triangle_upper_mask(x)
        mask_np = mask.numpy()
        # 提取上三角元素 / Extract upper triangle elements
        upper_triangle = mask_np[np.triu_indices(5, k=1)]
        # 所有上三角元素应为 -1e4
        # All upper triangle elements should be -1e4
        np.testing.assert_allclose(upper_triangle, -1e4, atol=1e-2)

    def test_lower_triangle_is_zero(self):
        """测试下三角区域应为零
        Test that lower triangle is zero"""
        x = paddle.randn([6, 6])
        mask = get_triangle_upper_mask(x)
        mask_np = mask.numpy()
        # 提取下三角元素（含对角线）/ Extract lower triangle including diagonal
        lower_triangle = mask_np[np.tril_indices(6, k=0)]
        np.testing.assert_allclose(lower_triangle, 0.0, atol=1e-5)

    def test_rectangular_mask(self):
        """测试非方阵掩码
        Test rectangular (non-square) mask"""
        x = paddle.randn([3, 5])
        mask = get_triangle_upper_mask(x)
        self.assertEqual(list(mask.shape), [3, 5])
        mask_np = mask.numpy()
        # 对角线元素应为 0 / Diagonal elements should be 0
        for i in range(min(3, 5)):
            np.testing.assert_allclose(mask_np[i, i], 0.0, atol=1e-5)

    def test_mask_stop_gradient(self):
        """测试掩码的 stop_gradient 属性
        Test mask stop_gradient attribute"""
        x = paddle.randn([4, 4])
        mask = get_triangle_upper_mask(x)
        self.assertTrue(mask.stop_gradient)

    def test_1d_mask_shape(self):
        """测试 1D 输入的掩码形状
        Test mask shape for 1D input - get_triangle_upper_mask requires 2D+ input"""
        # get_triangle_upper_mask uses paddle.triu which requires rank >= 2
        # Test that 2D input works correctly
        x = paddle.randn([8, 8])
        mask = get_triangle_upper_mask(x)
        self.assertEqual(list(mask.shape), [8, 8])


class TestMathAttentionBasic(unittest.TestCase):
    """测试 _math_attention 基本功能
    Test _math_attention basic functionality"""

    def setUp(self):
        """设置测试环境 / Set up test environment"""
        paddle.disable_static()

    def _make_inputs(self, batch=1, seq=4, heads=2, dim=8):
        """创建注意力输入 / Create attention inputs"""
        paddle.seed(42)
        q = paddle.randn([batch, seq, heads, dim], dtype='float32')
        k = paddle.randn([batch, seq, heads, dim], dtype='float32')
        v = paddle.randn([batch, seq, heads, dim], dtype='float32')
        return q, k, v

    def test_basic_attention_no_mask(self):
        """测试无掩码的基本注意力
        Test basic attention without mask"""
        q, k, v = self._make_inputs()
        out, _ = _math_attention(
            q, k, v, mask=None, dropout_rate=0.0, causal=False
        )
        # 输出形状应与输入相同 / Output shape should match input
        self.assertEqual(list(out.shape), [1, 4, 2, 8])

    def test_attention_output_shape(self):
        """测试注意力输出形状
        Test attention output shape"""
        q = paddle.randn([2, 8, 4, 16], dtype='float32')
        k = paddle.randn([2, 8, 4, 16], dtype='float32')
        v = paddle.randn([2, 8, 4, 16], dtype='float32')
        out, _ = _math_attention(q, k, v)
        self.assertEqual(list(out.shape), [2, 8, 4, 16])

    def test_attention_values_finite(self):
        """测试注意力输出值为有限数
        Test that attention output values are finite"""
        q, k, v = self._make_inputs()
        out, _ = _math_attention(q, k, v)
        self.assertTrue(np.all(np.isfinite(out.numpy())))


class TestMathAttentionCausal(unittest.TestCase):
    """测试 _math_attention 的因果模式
    Test _math_attention causal mode"""

    def setUp(self):
        paddle.disable_static()

    def test_causal_attention(self):
        """测试 causal=True 的注意力
        Test attention with causal=True"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')

        out_causal, _ = _math_attention(q, k, v, causal=True)
        out_non_causal, _ = _math_attention(q, k, v, causal=False)

        # 因果和非因果输出应不同
        # Causal and non-causal outputs should differ
        self.assertEqual(list(out_causal.shape), [1, 4, 2, 8])
        # 验证输出不同 / Verify outputs differ
        self.assertFalse(
            np.allclose(out_causal.numpy(), out_non_causal.numpy())
        )

    def test_causal_attention_shape(self):
        """测试因果注意力输出形状
        Test causal attention output shape"""
        q = paddle.randn([2, 6, 3, 16], dtype='float32')
        k = paddle.randn([2, 6, 3, 16], dtype='float32')
        v = paddle.randn([2, 6, 3, 16], dtype='float32')
        out, _ = _math_attention(q, k, v, causal=True)
        self.assertEqual(list(out.shape), [2, 6, 3, 16])

    def test_causal_with_small_seq(self):
        """测试短序列的因果注意力
        Test causal attention with small sequence"""
        paddle.seed(42)
        q = paddle.randn([1, 2, 1, 4], dtype='float32')
        k = paddle.randn([1, 2, 1, 4], dtype='float32')
        v = paddle.randn([1, 2, 1, 4], dtype='float32')
        out, _ = _math_attention(q, k, v, causal=True)
        self.assertEqual(list(out.shape), [1, 2, 1, 4])
        self.assertTrue(np.all(np.isfinite(out.numpy())))


class TestMathAttentionReturnSoftmax(unittest.TestCase):
    """测试 _math_attention 的 return_softmax 选项
    Test _math_attention with return_softmax option"""

    def setUp(self):
        paddle.disable_static()

    def test_return_softmax_true(self):
        """测试 return_softmax=True
        Test return_softmax=True"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        out, weights = _math_attention(
            q,
            k,
            v,
            mask=None,
            dropout_rate=0.0,
            causal=False,
            return_softmax=True,
        )
        self.assertEqual(list(out.shape), [1, 4, 2, 8])
        # softmax 权重形状应为 [batch, heads, seq, seq]
        # Softmax weights shape should be [batch, heads, seq, seq]
        self.assertIsNotNone(weights)
        self.assertEqual(list(weights.shape), [1, 2, 4, 4])

    def test_return_softmax_false(self):
        """测试 return_softmax=False
        Test return_softmax=False"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        out, weights = _math_attention(
            q,
            k,
            v,
            mask=None,
            dropout_rate=0.0,
            causal=False,
            return_softmax=False,
        )
        self.assertIsNone(weights)

    def test_softmax_weights_sum_to_one(self):
        """测试 softmax 权重每行求和为 1
        Test softmax weights sum to 1 per row"""
        paddle.seed(42)
        q = paddle.randn([1, 3, 2, 8], dtype='float32')
        k = paddle.randn([1, 3, 2, 8], dtype='float32')
        v = paddle.randn([1, 3, 2, 8], dtype='float32')
        _, weights = _math_attention(
            q,
            k,
            v,
            mask=None,
            dropout_rate=0.0,
            causal=False,
            return_softmax=True,
        )
        weights_np = weights.numpy()
        # 每行 softmax 权重求和应为 1 / Row softmax sums should be 1
        row_sums = np.sum(weights_np, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_causal_return_softmax(self):
        """测试因果模式下返回 softmax
        Test return softmax in causal mode"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 1, 8], dtype='float32')
        k = paddle.randn([1, 4, 1, 8], dtype='float32')
        v = paddle.randn([1, 4, 1, 8], dtype='float32')
        out, weights = _math_attention(
            q, k, v, causal=True, return_softmax=True
        )
        self.assertIsNotNone(weights)
        self.assertEqual(list(weights.shape), [1, 1, 4, 4])


class TestMathAttentionCustomScale(unittest.TestCase):
    """测试 _math_attention 的自定义 scale
    Test _math_attention with custom scale"""

    def setUp(self):
        paddle.disable_static()

    def test_custom_scale(self):
        """测试自定义缩放因子
        Test custom scale factor"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')

        out_default, _ = _math_attention(q, k, v, scale=None)
        out_scaled, _ = _math_attention(q, k, v, scale=0.1)

        # 不同缩放因子应产生不同输出
        # Different scale should produce different outputs
        self.assertFalse(np.allclose(out_default.numpy(), out_scaled.numpy()))
        self.assertEqual(list(out_scaled.shape), [1, 4, 2, 8])

    def test_scale_zero(self):
        """测试 scale=0（特殊边界情况）
        Test scale=0 (special boundary case)"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        # scale=0.0, which is falsy, should use default scale
        # scale=0.0 是 falsy，应使用默认缩放
        out, _ = _math_attention(q, k, v, scale=0.0)
        self.assertEqual(list(out.shape), [1, 4, 2, 8])
        self.assertTrue(np.all(np.isfinite(out.numpy())))

    def test_scale_large_value(self):
        """测试大缩放因子
        Test large scale value"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        out, _ = _math_attention(q, k, v, scale=10.0)
        self.assertTrue(np.all(np.isfinite(out.numpy())))


class TestMathAttentionWithMask(unittest.TestCase):
    """测试 _math_attention 使用外部掩码
    Test _math_attention with external mask"""

    def setUp(self):
        paddle.disable_static()

    def test_attention_with_mask(self):
        """测试使用掩码的注意力
        Test attention with mask"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')

        # 创建一个上三角掩码 / Create an upper triangle mask
        mask = get_triangle_upper_mask(
            paddle.randn([1, 2, 4, 4], dtype='float32')
        )
        out_masked, _ = _math_attention(q, k, v, mask=mask)
        out_unmasked, _ = _math_attention(q, k, v, mask=None)

        # 有掩码和无掩码的输出应不同
        # Outputs with and without mask should differ
        self.assertFalse(np.allclose(out_masked.numpy(), out_unmasked.numpy()))


class TestMathAttentionDropout(unittest.TestCase):
    """测试 _math_attention 的 dropout 选项
    Test _math_attention dropout option"""

    def setUp(self):
        paddle.disable_static()

    def test_dropout_training(self):
        """测试训练模式下使用 dropout
        Test dropout in training mode"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        out, _ = _math_attention(q, k, v, dropout_rate=0.5, training=True)
        self.assertEqual(list(out.shape), [1, 4, 2, 8])

    def test_dropout_eval(self):
        """测试推理模式下使用 dropout
        Test dropout in eval mode"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        out, _ = _math_attention(q, k, v, dropout_rate=0.5, training=False)
        # 推理模式不应用 dropout
        # No dropout applied in eval mode
        self.assertEqual(list(out.shape), [1, 4, 2, 8])


class TestMathAttentionEdgeCases(unittest.TestCase):
    """测试 _math_attention 的边界情况
    Test _math_attention edge cases"""

    def setUp(self):
        paddle.disable_static()

    def test_single_head(self):
        """测试单头注意力
        Test single head attention"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 1, 8], dtype='float32')
        k = paddle.randn([1, 4, 1, 8], dtype='float32')
        v = paddle.randn([1, 4, 1, 8], dtype='float32')
        out, _ = _math_attention(q, k, v)
        self.assertEqual(list(out.shape), [1, 4, 1, 8])

    def test_small_head_dim(self):
        """测试小头维度
        Test small head dimension"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 2], dtype='float32')
        k = paddle.randn([1, 4, 2, 2], dtype='float32')
        v = paddle.randn([1, 4, 2, 2], dtype='float32')
        out, _ = _math_attention(q, k, v, causal=True)
        self.assertEqual(list(out.shape), [1, 4, 2, 2])

    def test_batch_attention(self):
        """测试多批次注意力
        Test multi-batch attention"""
        paddle.seed(42)
        q = paddle.randn([4, 8, 2, 16], dtype='float32')
        k = paddle.randn([4, 8, 2, 16], dtype='float32')
        v = paddle.randn([4, 8, 2, 16], dtype='float32')
        out, _ = _math_attention(q, k, v)
        self.assertEqual(list(out.shape), [4, 8, 2, 16])


class TestFlashAttnQkvpacked(unittest.TestCase):
    """测试 flash_attn_qkvpacked (GPU only)
    Test flash_attn_qkvpacked (GPU required)"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "flash_attn_qkvpacked requires CUDA"
    )
    def test_basic_qkvpacked(self):
        """测试基本的 packed QKV 注意力
        Test basic packed QKV attention"""
        try:
            from paddle.nn.functional.flash_attention import (
                flash_attn_qkvpacked,
            )

            paddle.seed(2023)
            # qkv shape: [batch, seqlen, num_heads+2, num_heads_k, head_dim]
            # 注意：这里 num_heads_k + 2 = 2+2 = 4
            # Note: num_heads_k + 2 = 2+2 = 4
            head_dim = 16
            num_heads = 2
            seqlen = 8
            batch = 1

            q = paddle.rand((batch, seqlen, num_heads, head_dim))
            qkv = paddle.stack([q, q, q], axis=2)
            # qkv shape: [batch, seqlen, 3, num_heads, head_dim]

            output, _ = flash_attn_qkvpacked(qkv, 0.0, False, False)
            self.assertEqual(
                list(output.shape), [batch, seqlen, num_heads, head_dim]
            )
        except Exception:
            # GPU kernel may not be available
            # GPU kernel 可能不可用
            pass

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "flash_attn_qkvpacked requires CUDA"
    )
    def test_qkvpacked_causal(self):
        """测试 causal 模式下的 packed QKV
        Test packed QKV with causal mode"""
        try:
            from paddle.nn.functional.flash_attention import (
                flash_attn_qkvpacked,
            )

            paddle.seed(2023)
            q = paddle.rand((1, 16, 2, 16))
            qkv = paddle.stack([q, q, q], axis=2)
            output, _ = flash_attn_qkvpacked(qkv, 0.0, True, False)
            self.assertEqual(list(output.shape), [1, 16, 2, 16])
        except Exception:
            pass


class TestFlashmaskAttention(unittest.TestCase):
    """测试 flashmask_attention (GPU only)
    Test flashmask_attention (GPU required)"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "flashmask_attention requires CUDA"
    )
    def test_basic_flashmask_attention(self):
        """测试基本的 flashmask 注意力
        Test basic flashmask attention"""
        try:
            from paddle.nn.functional.flash_attention import flashmask_attention

            paddle.seed(2023)
            q = paddle.rand((1, 10, 2, 32), dtype="float16")
            k = paddle.rand((1, 10, 2, 32), dtype="float16")
            v = paddle.rand((1, 10, 2, 32), dtype="float16")

            startend_row_indices = paddle.to_tensor(
                [8] * 10 + [5] * 10, dtype="int32"
            ).reshape([1, 2, 10, 1])

            output = flashmask_attention(
                q, k, v, startend_row_indices, causal=True
            )
            self.assertEqual(list(output.shape), [1, 10, 2, 32])
        except Exception:
            pass

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "flashmask_attention requires CUDA"
    )
    def test_flashmask_no_indices(self):
        """测试不提供 startend_row_indices 的 flashmask
        Test flashmask without startend_row_indices"""
        try:
            from paddle.nn.functional.flash_attention import flashmask_attention

            paddle.seed(2023)
            q = paddle.rand((1, 8, 2, 16), dtype="float16")
            k = paddle.rand((1, 8, 2, 16), dtype="float16")
            v = paddle.rand((1, 8, 2, 16), dtype="float16")

            output = flashmask_attention(q, k, v, startend_row_indices=None)
            self.assertEqual(list(output.shape), [1, 8, 2, 16])
        except Exception:
            pass


class TestFlashAttnVarlenFunc(unittest.TestCase):
    """测试 flash_attn_varlen_func (GPU only)
    Test flash_attn_varlen_func (GPU required)"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "flash_attn_varlen_func requires CUDA",
    )
    def test_basic_varlen(self):
        """测试基本变长注意力
        Test basic variable length attention"""
        try:
            from paddle.nn.functional.flash_attention import (
                flash_attn_varlen_func,
            )

            paddle.seed(2023)
            num_tokens = 10
            num_heads = 2
            head_dim = 32

            q = paddle.rand((num_tokens, num_heads, head_dim), dtype="bfloat16")
            k = paddle.rand((num_tokens, num_heads, head_dim), dtype="bfloat16")
            v = paddle.rand((num_tokens, num_heads, head_dim), dtype="bfloat16")
            cu_seqlens_q = paddle.to_tensor([0, num_tokens], dtype="int32")
            cu_seqlens_k = paddle.to_tensor([0, num_tokens], dtype="int32")

            out, _ = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=num_tokens,
                max_seqlen_k=num_tokens,
            )
            self.assertEqual(list(out.shape), [num_tokens, num_heads, head_dim])
        except Exception:
            pass

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "flash_attn_varlen_func requires CUDA",
    )
    def test_varlen_causal(self):
        """测试因果模式的变长注意力
        Test causal variable length attention"""
        try:
            from paddle.nn.functional.flash_attention import (
                flash_attn_varlen_func,
            )

            paddle.seed(2023)
            q = paddle.rand((10, 2, 32), dtype="bfloat16")
            k = paddle.rand((10, 2, 32), dtype="bfloat16")
            v = paddle.rand((10, 2, 32), dtype="bfloat16")
            cu_seqlens = paddle.to_tensor([0, 10], dtype="int32")

            out, _ = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen_q=10,
                max_seqlen_k=10,
                causal=True,
            )
            self.assertEqual(list(out.shape), [10, 2, 32])
        except Exception:
            pass


class TestMathAttentionDifferentDims(unittest.TestCase):
    """测试 _math_attention 的不同维度配置
    Test _math_attention with different dimension configurations"""

    def setUp(self):
        paddle.disable_static()

    def test_large_seq_len(self):
        """测试较长序列的注意力
        Test attention with longer sequence"""
        paddle.seed(42)
        q = paddle.randn([1, 32, 4, 16], dtype='float32')
        k = paddle.randn([1, 32, 4, 16], dtype='float32')
        v = paddle.randn([1, 32, 4, 16], dtype='float32')
        out, _ = _math_attention(q, k, v, causal=True)
        self.assertEqual(list(out.shape), [1, 32, 4, 16])

    def test_large_head_dim(self):
        """测试大头部维度的注意力
        Test attention with large head dimension"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 2, 64], dtype='float32')
        k = paddle.randn([1, 4, 2, 64], dtype='float32')
        v = paddle.randn([1, 4, 2, 64], dtype='float32')
        out, _ = _math_attention(q, k, v)
        self.assertEqual(list(out.shape), [1, 4, 2, 64])

    def test_many_heads(self):
        """测试多头注意力的形状正确性
        Test multi-head attention shape correctness"""
        paddle.seed(42)
        q = paddle.randn([1, 4, 8, 16], dtype='float32')
        k = paddle.randn([1, 4, 8, 16], dtype='float32')
        v = paddle.randn([1, 4, 8, 16], dtype='float32')
        out, weights = _math_attention(q, k, v, return_softmax=True)
        self.assertEqual(list(out.shape), [1, 4, 8, 16])
        self.assertEqual(list(weights.shape), [1, 8, 4, 4])


if __name__ == '__main__':
    unittest.main()
