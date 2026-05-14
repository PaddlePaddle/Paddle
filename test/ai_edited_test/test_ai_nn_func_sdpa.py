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

# [AUTO-GENERATED]
# Target: paddle/nn/functional/sdpa.py
# Coverage target: improve coverage for scaled_dot_product_attention,
#   SDPParams, _repeat_kv, and related helper functions
"""
Tests for paddle.nn.functional.sdpa module.
测试 paddle.nn.functional.sdpa 模块的单元测试。
"""

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(),
    "CUDA is required for scaled_dot_product_attention GPU tests",
)
class TestScaledDotProductAttentionCUDA(unittest.TestCase):
    """Tests for scaled_dot_product_attention on CUDA. / CUDA 上 scaled_dot_product_attention 的测试。"""

    def setUp(self):
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 8
        self.head_dim = 16

    def _make_inputs(self, dtype='float32'):
        """Helper to create Q, K, V inputs. / 创建 Q、K、V 输入的辅助方法。"""
        shape = [self.batch_size, self.seq_len, self.num_heads, self.head_dim]
        q = paddle.randn(shape, dtype=dtype)
        k = paddle.randn(shape, dtype=dtype)
        v = paddle.randn(shape, dtype=dtype)
        return q, k, v

    def test_sdpa_basic_float32(self):
        """Test SDPA with float32 inputs. / 测试 float32 输入的 SDPA。"""
        q, k, v = self._make_inputs('float32')
        out = F.scaled_dot_product_attention(q, k, v)
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_with_dropout(self):
        """Test SDPA with dropout. / 测试带 dropout 的 SDPA。"""
        q, k, v = self._make_inputs('float32')
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.1, training=True
        )
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_causal(self):
        """Test SDPA with causal mask. / 测试因果掩码的 SDPA。"""
        q, k, v = self._make_inputs('float32')
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_with_attn_mask_float(self):
        """Test SDPA with float attention mask. / 测试浮点注意力掩码的 SDPA。"""
        q, k, v = self._make_inputs('float32')
        mask = paddle.randn(
            [self.batch_size, self.num_heads, self.seq_len, self.seq_len],
            dtype='float32',
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_with_bool_mask(self):
        """Test SDPA with bool attention mask. / 测试布尔注意力掩码的 SDPA。"""
        q, k, v = self._make_inputs('float32')
        mask = paddle.ones(
            [self.batch_size, self.num_heads, self.seq_len, self.seq_len],
            dtype='bool',
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_3d_input(self):
        """Test SDPA with 3D (unbatched) input. / 测试三维（无批次）输入的 SDPA。"""
        shape = [self.seq_len, self.num_heads, self.head_dim]
        q = paddle.randn(shape, dtype='float32')
        k = paddle.randn(shape, dtype='float32')
        v = paddle.randn(shape, dtype='float32')
        out = F.scaled_dot_product_attention(q, k, v)
        self.assertEqual(
            out.shape, [self.seq_len, self.num_heads, self.head_dim]
        )

    def test_sdpa_eval_mode(self):
        """Test SDPA in eval mode. / 测试评估模式的 SDPA。"""
        q, k, v = self._make_inputs('float32')
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.5, training=False
        )
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_float16(self):
        """Test SDPA with float16 inputs. / 测试 float16 输入的 SDPA。"""
        q, k, v = self._make_inputs('float16')
        out = F.scaled_dot_product_attention(q, k, v)
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_gqa(self):
        """Test SDPA with Group Query Attention. / 测试分组查询注意力的 SDPA。"""
        num_kv_heads = 2
        q_shape = [self.batch_size, self.seq_len, self.num_heads, self.head_dim]
        kv_shape = [self.batch_size, self.seq_len, num_kv_heads, self.head_dim]
        q = paddle.randn(q_shape, dtype='float32')
        k = paddle.randn(kv_shape, dtype='float32')
        v = paddle.randn(kv_shape, dtype='float32')
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        self.assertEqual(out.shape, q_shape)

    def test_sdpa_gqa_disabled(self):
        """Test SDPA with GQA disabled (equal heads). / 测试禁用 GQA 的 SDPA（等头数）。"""
        q, k, v = self._make_inputs('float32')
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=False)
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_2d_mask(self):
        """Test SDPA with 2D attention mask. / 测试二维注意力掩码的 SDPA。"""
        q, k, v = self._make_inputs('float32')
        mask = paddle.randn([self.seq_len, self.seq_len], dtype='float32')
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_sdpa_3d_mask(self):
        """Test SDPA with 3D attention mask. / 测试三维注意力掩码的 SDPA。"""
        q, k, v = self._make_inputs('float32')
        mask = paddle.randn(
            [self.batch_size, self.seq_len, self.seq_len], dtype='float32'
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )


class TestRepeatKV(unittest.TestCase):
    """Tests for _repeat_kv helper function. / _repeat_kv 辅助函数的测试。"""

    def test_repeat_kv_no_repeat(self):
        """Test _repeat_kv with num_repeats=1. / 测试 num_repeats=1 的 _repeat_kv。"""
        from paddle.nn.functional.sdpa import _repeat_kv

        key = paddle.randn([2, 8, 4, 16], dtype='float32')
        value = paddle.randn([2, 8, 4, 16], dtype='float32')
        k_out, v_out = _repeat_kv(key, value, 1)
        np.testing.assert_array_equal(k_out.numpy(), key.numpy())
        np.testing.assert_array_equal(v_out.numpy(), value.numpy())

    def test_repeat_kv_with_repeat(self):
        """Test _repeat_kv with num_repeats=2. / 测试 num_repeats=2 的 _repeat_kv。"""
        from paddle.nn.functional.sdpa import _repeat_kv

        key = paddle.randn([2, 8, 2, 16], dtype='float32')
        value = paddle.randn([2, 8, 2, 16], dtype='float32')
        k_out, v_out = _repeat_kv(key, value, 2)
        self.assertEqual(k_out.shape, [2, 8, 4, 16])
        self.assertEqual(v_out.shape, [2, 8, 4, 16])


class TestSDPParams(unittest.TestCase):
    """Tests for SDPParams dataclass. / SDPParams 数据类的测试。"""

    def test_sdp_params_batch_size(self):
        """Test SDPParams batch_size cached property. / 测试 SDPParams batch_size 缓存属性。"""
        from paddle.nn.functional.sdpa import SDPParams

        params = SDPParams(
            query_shape=paddle.Size([2, 8, 4, 16]),
            key_shape=paddle.Size([2, 8, 4, 16]),
            value_shape=paddle.Size([2, 8, 4, 16]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=None,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        self.assertEqual(params.batch_size, (2, 2, 2))

    def test_sdp_params_seq_len(self):
        """Test SDPParams seq_len cached property. / 测试 SDPParams seq_len 缓存属性。"""
        from paddle.nn.functional.sdpa import SDPParams

        params = SDPParams(
            query_shape=paddle.Size([2, 8, 4, 16]),
            key_shape=paddle.Size([2, 6, 4, 16]),
            value_shape=paddle.Size([2, 6, 4, 16]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=None,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        self.assertEqual(params.seq_len, (8, 6, 6))

    def test_sdp_params_num_heads(self):
        """Test SDPParams num_heads cached property. / 测试 SDPParams num_heads 缓存属性。"""
        from paddle.nn.functional.sdpa import SDPParams

        params = SDPParams(
            query_shape=paddle.Size([2, 8, 4, 16]),
            key_shape=paddle.Size([2, 8, 2, 16]),
            value_shape=paddle.Size([2, 8, 2, 16]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=None,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        self.assertEqual(params.num_heads, (4, 2, 2))

    def test_sdp_params_head_dim(self):
        """Test SDPParams head_dim cached property. / 测试 SDPParams head_dim 缓存属性。"""
        from paddle.nn.functional.sdpa import SDPParams

        params = SDPParams(
            query_shape=paddle.Size([2, 8, 4, 16]),
            key_shape=paddle.Size([2, 8, 4, 32]),
            value_shape=paddle.Size([2, 8, 4, 32]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=None,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        self.assertEqual(params.head_dim, (16, 32, 32))


class TestSDPAHelpers(unittest.TestCase):
    """Tests for SDPA helper functions. / SDPA 辅助函数的测试。"""

    def test_get_device_capability_cpu(self):
        """Test get_device_capability for CPU (device_id < 0). / 测试 CPU 的 get_device_capability。"""
        from paddle.nn.functional.sdpa import get_device_capability

        result = get_device_capability(-1)
        self.assertEqual(result, (0, 0))

    def test_check_cuda_is_available(self):
        """Test check_cuda_is_available returns bool. / 测试 check_cuda_is_available 返回布尔值。"""
        from paddle.nn.functional.sdpa import check_cuda_is_available

        result = check_cuda_is_available()
        self.assertIsInstance(result, bool)

    def test_check_sm_version(self):
        """Test check_sm_version returns bool. / 测试 check_sm_version 返回布尔值。"""
        from paddle.nn.functional.sdpa import check_sm_version

        result = check_sm_version((8, 0), (12, 1), 0)
        self.assertIsInstance(result, bool)

    def test_check_all_tensors_on_device_cpu(self):
        """Test check_all_tensors_on_device returns False for CPU. / 测试 CPU 的 check_all_tensors_on_device 返回 False。"""
        from paddle.nn.functional.sdpa import (
            SDPParams,
            check_all_tensors_on_device,
        )

        params = SDPParams(
            query_shape=paddle.Size([1, 8, 4, 16]),
            key_shape=paddle.Size([1, 8, 4, 16]),
            value_shape=paddle.Size([1, 8, 4, 16]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=None,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        result = check_all_tensors_on_device(params)
        self.assertFalse(result)

    def test_check_head_dim_size_flash(self):
        """Test check_head_dim_size_flash returns False for mismatched dims. / 测试不匹配维度的 check_head_dim_size_flash 返回 False。"""
        from paddle.nn.functional.sdpa import (
            SDPParams,
            check_head_dim_size_flash,
        )

        params = SDPParams(
            query_shape=paddle.Size([1, 8, 4, 16]),
            key_shape=paddle.Size([1, 8, 4, 32]),
            value_shape=paddle.Size([1, 8, 4, 32]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=None,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        result = check_head_dim_size_flash(params)
        self.assertFalse(result)

    def test_check_scale_is_none(self):
        """Test check_scale_is_none returns True when scale is None. / 测试 scale 为 None 时 check_scale_is_none 返回 True。"""
        from paddle.nn.functional.sdpa import SDPParams, check_scale_is_None

        params = SDPParams(
            query_shape=paddle.Size([1, 8, 4, 16]),
            key_shape=paddle.Size([1, 8, 4, 16]),
            value_shape=paddle.Size([1, 8, 4, 16]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=None,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        result = check_scale_is_None(params)
        self.assertTrue(result)

    def test_check_scale_is_not_none(self):
        """Test check_scale_is_none returns False when scale is set. / 测试 scale 被设置时 check_scale_is_none 返回 False。"""
        from paddle.nn.functional.sdpa import SDPParams, check_scale_is_None

        params = SDPParams(
            query_shape=paddle.Size([1, 8, 4, 16]),
            key_shape=paddle.Size([1, 8, 4, 16]),
            value_shape=paddle.Size([1, 8, 4, 16]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=0.5,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        result = check_scale_is_None(params)
        self.assertFalse(result)

    def test_check_flash_causal_non_square_seqlens(self):
        """Test check_flash_causal_non_square_seqlens returns True when non-causal. / 测试非因果时返回 True。"""
        from paddle.nn.functional.sdpa import (
            SDPParams,
            check_flash_causal_non_square_seqlens,
        )

        params = SDPParams(
            query_shape=paddle.Size([1, 8, 4, 16]),
            key_shape=paddle.Size([1, 6, 4, 16]),
            value_shape=paddle.Size([1, 6, 4, 16]),
            attn_mask_shape=None,
            dropout=0.0,
            is_causal=False,
            scale=None,
            query_stop_gradient=True,
            dtype=(paddle.float32, paddle.float32, paddle.float32),
            place=(paddle.CPUPlace(), paddle.CPUPlace(), paddle.CPUPlace()),
        )
        result = check_flash_causal_non_square_seqlens(params)
        self.assertTrue(result)


class TestScaledDotProductAttentionCPU(unittest.TestCase):
    """Tests for SDPA on CPU (fallback to math backend). / CPU 上 SDPA 的测试（回退到 math 后端）。"""

    def test_sdpa_cpu_basic(self):
        """Test SDPA on CPU falls back to math. / 测试 CPU 上 SDPA 回退到 math。"""
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        out = F.scaled_dot_product_attention(q, k, v)
        self.assertEqual(out.shape, [1, 4, 2, 8])

    def test_sdpa_cpu_3d(self):
        """Test SDPA on CPU with 3D input. / 测试 CPU 上三维输入的 SDPA。"""
        q = paddle.randn([4, 2, 8], dtype='float32')
        k = paddle.randn([4, 2, 8], dtype='float32')
        v = paddle.randn([4, 2, 8], dtype='float32')
        out = F.scaled_dot_product_attention(q, k, v)
        self.assertEqual(out.shape, [4, 2, 8])

    def test_sdpa_cpu_causal(self):
        """Test SDPA on CPU with causal. / 测试 CPU 上因果掩码的 SDPA。"""
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(out.shape, [1, 4, 2, 8])

    def test_sdpa_cpu_with_mask(self):
        """Test SDPA on CPU with attention mask. / 测试 CPU 上带注意力掩码的 SDPA。"""
        q = paddle.randn([1, 4, 2, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        mask = paddle.ones([1, 1, 4, 4], dtype='bool')
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(out.shape, [1, 4, 2, 8])

    def test_sdpa_cpu_gqa(self):
        """Test SDPA on CPU with GQA. / 测试 CPU 上 GQA 的 SDPA。"""
        q = paddle.randn([1, 4, 4, 8], dtype='float32')
        k = paddle.randn([1, 4, 2, 8], dtype='float32')
        v = paddle.randn([1, 4, 2, 8], dtype='float32')
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        self.assertEqual(out.shape, [1, 4, 4, 8])


if __name__ == '__main__':
    unittest.main()
