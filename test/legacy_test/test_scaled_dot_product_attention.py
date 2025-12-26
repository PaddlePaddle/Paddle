# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from op_test import get_device_place, is_custom_device

import paddle
import paddle.nn.functional as F
from paddle.nn.functional import (
    scaled_dot_product_attention,
    sdp_kernel,
)

fa_available = paddle.framework._global_flags().get(
    "FLAGS_flash_attn_available", False
)
mea_available = paddle.framework._global_flags().get(
    "FLAGS_mem_efficient_attn_available", False
)


def attention_naive(q, k, v, causal=False):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt * scale, paddle.transpose(kt, [0, 1, 3, 2]))
    p = (
        paddle.incubate.softmax_mask_fuse_upper_triangle(s)
        if causal
        else F.softmax(s)
    )
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


def attention_naive_with_mask(q, k, v, attn_bias):
    if attn_bias is not None and attn_bias.dim() == 3:
        attn_bias = paddle.unsqueeze(attn_bias, axis=1)
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = F.softmax(s + attn_bias)
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


def attention_naive_with_bool_mask(q, k, v, bool_mask):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])

    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)

    float_mask = paddle.where(
        bool_mask,
        paddle.to_tensor(0.0, dtype=q.dtype),
        paddle.to_tensor(-float('inf'), dtype=q.dtype),
    )

    s = s + float_mask
    p = F.softmax(s)

    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


@unittest.skipIf(
    not (paddle.is_compiled_with_cuda() or is_custom_device())
    or paddle.is_compiled_with_rocm(),
    "CUDA is not available, this test requires GPU support.",
)
class TestAttentionWithBoolMask(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (1, 8, 8, 8)
        self.dtype = 'float32'
        self.dropout = 0.0
        self.causal = False

    def test_dot_scale_product_bool_mask(self):
        # test dynamic
        paddle.disable_static()

        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        mask_shape = (self.shape[0], 1, self.shape[1], self.shape[1])
        bool_mask = np.random.choice([True, False], size=mask_shape)

        m = paddle.to_tensor(
            bool_mask, place=self.place, dtype=paddle.bool, stop_gradient=False
        )

        with sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False
        ):
            out = scaled_dot_product_attention(
                q, k, v, m, self.dropout, self.causal
            )

        out_ = attention_naive_with_bool_mask(q_, k_, v_, m)

        out.backward()
        out_.backward()

        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

    def test_dot_scale_product_float_mask(self):
        # test with mask=float
        paddle.disable_static()

        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        mask_shape = (self.shape[0], 1, self.shape[1], self.shape[1])
        mask = np.random.random(mask_shape)
        m = paddle.to_tensor(
            mask, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        with sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False
        ):
            out = scaled_dot_product_attention(
                q, k, v, m, self.dropout, self.causal
            )

        out_ = attention_naive_with_mask(q_, k_, v_, m)
        out.backward()
        out_.backward()
        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

    def test_efficient_backend_with_mask(self):
        """
        Test efficient backend selection when mask is present.
        """
        paddle.disable_static()
        query = np.random.random(self.shape).astype(self.dtype)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        mask_shape = (self.shape[0], 1, self.shape[1], self.shape[1])
        mask = np.random.random(mask_shape).astype(self.dtype)
        m = paddle.to_tensor(
            mask, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        # Enable only efficient backend
        with sdp_kernel(
            enable_math=False, enable_flash=False, enable_mem_efficient=True
        ):
            # This will enter _select_sdp_for_sdpa, check EFFICIENT_ATTENTION,
            # pass can_use_efficient, and return "mem_efficient"
            out = scaled_dot_product_attention(
                q, q, q, m, self.dropout, self.causal
            )

        # Compare with naive math implementation for correctness
        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        out_ = attention_naive_with_mask(q_, q_, q_, m)
        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

    def test_flash_backend_rejection(self):
        """
        Test that flash backend is skipped and RuntimeError is raised
        if conditions are not met (e.g., head_dim > 256), regardless of hardware.
        """
        paddle.disable_static()

        # Use head_dim = 288, which is > 256
        # This will *always* fail can_use_flash_attn()
        shape = (1, 8, 2, 288)
        dtype = 'float16'

        query = np.random.random(shape).astype(dtype)
        q = paddle.to_tensor(
            query, place=self.place, dtype=dtype, stop_gradient=False
        )

        mask_shape = (shape[0], 1, shape[1], shape[1])
        mask = np.random.random(mask_shape).astype(dtype)
        m = paddle.to_tensor(
            mask, place=self.place, dtype=dtype, stop_gradient=False
        )

        # Enable *only* flash backend
        with (
            sdp_kernel(
                enable_math=False, enable_flash=True, enable_mem_efficient=False
            ),
            self.assertRaises(
                RuntimeError,
                msg="No available backend for scaled_dot_product_attention was found.",
            ),
        ):
            _ = scaled_dot_product_attention(
                q, q, q, m, self.dropout, self.causal
            )


class TestAttentionWith3DInput(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float32'
        self.dropout = 0.0
        self.causal = False

    def test_3d_input(self):
        """Test scaled_dot_product_attention with 3D input tensors."""
        # test dynamic
        paddle.disable_static()

        shape_3d = (8, 1, 8)

        query = np.random.random(shape_3d).astype(np.float32)
        key = np.random.random(shape_3d).astype(np.float32)
        value = np.random.random(shape_3d).astype(np.float32)

        q = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        k = paddle.to_tensor(key, dtype=self.dtype, stop_gradient=False)
        v = paddle.to_tensor(value, dtype=self.dtype, stop_gradient=False)

        q_ref = paddle.unsqueeze(q, axis=0)
        k_ref = paddle.unsqueeze(k, axis=0)
        v_ref = paddle.unsqueeze(v, axis=0)

        with sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False
        ):
            out = scaled_dot_product_attention(
                q, k, v, None, self.dropout, self.causal
            )

        out_ref = attention_naive(q_ref, k_ref, v_ref, self.causal)

        out_ref = paddle.squeeze(out_ref, axis=0)

        np.testing.assert_allclose(out.numpy(), out_ref, rtol=5e-03, atol=1e-03)


class TestAttentionWithBoolMaskZeroSize(TestAttentionWithBoolMask):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (0, 8, 8, 8)
        self.dtype = 'float32'
        self.dropout = 0.0
        self.causal = False


class TestSDPKernelFlags(unittest.TestCase):
    def test_sdp_kernel_value_error(self):
        """
        Test ValueError when no backend is enabled in sdp_kernel.
        """
        with (
            self.assertRaises(
                ValueError, msg="At least one backend must be enabled"
            ),
            sdp_kernel(
                enable_math=False,
                enable_flash=False,
                enable_mem_efficient=False,
            ),
        ):
            pass

    def test_sdp_kernel_all_flags(self):
        """
        Test that sdp_kernel runs with flash and efficient flags.
        """
        # This test just ensures the context manager itself works
        # when flags are enabled.
        with sdp_kernel(
            enable_math=False,
            enable_flash=True,
            enable_mem_efficient=True,
        ):
            pass


@unittest.skipIf(
    not paddle.device.is_available(),
    "Skip test on CPU for cpu does not support fp16 matmul",
)
class TestZeroSizeBase(unittest.TestCase):
    def setUp(self) -> None:
        self.query_shape = [1, 0, 32, 128]
        self.key_shape = [1, 1024, 32, 128]
        self.value_shape = [1, 1024, 32, 128]
        self.attn_mask_shape = None
        self.is_causal = True
        self.expected_out_shape = [1, 0, 32, 128]
        self.dtype = 'float16'

    def prepare_input(self):
        kwargs = {
            "query": paddle.randn(shape=self.query_shape, dtype=self.dtype),
            "key": paddle.randn(shape=self.key_shape, dtype=self.dtype),
            "value": paddle.randn(shape=self.value_shape, dtype=self.dtype),
            "attn_mask": paddle.randn(
                shape=self.attn_mask_shape, dtype=self.dtype
            )
            if self.attn_mask_shape
            else None,
            "is_causal": self.is_causal,
        }
        return kwargs

    def test_dygraph(self):
        paddle.disable_static()
        kwargs = self.prepare_input()
        out = scaled_dot_product_attention(**kwargs)
        self.assertEqual(out.shape, self.expected_out_shape)

    def test_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            kwargs = self.prepare_input()
            out = scaled_dot_product_attention(**kwargs)
            exe = paddle.static.Executor()
            outs = exe.run(fetch_list=[out])
            self.assertEqual(list(outs[0].shape), self.expected_out_shape)
        paddle.disable_static()


class TestZeroSizeCase2(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [2, 0, 12, 64]
        self.key_shape = [2, 64, 12, 64]
        self.value_shape = [2, 64, 12, 64]
        self.attn_mask_shape = None
        self.is_causal = True
        self.expected_out_shape = [2, 0, 12, 64]
        self.dtype = 'float16'


class TestZeroSizeCase3(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [1, 2048, 8, 0]
        self.key_shape = [1, 2048, 2, 0]
        self.value_shape = [1, 2048, 2, 0]
        self.attn_mask_shape = [1, 1, 2048, 0]
        self.is_causal = True
        self.expected_out_shape = [1, 2048, 8, 0]
        self.dtype = 'float16'


class TestZeroSizeCase4(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [1, 2048, 8, 16]
        self.key_shape = [1, 2048, 2, 16]
        self.value_shape = [1, 2048, 2, 0]
        self.attn_mask_shape = [1, 1, 2048, 2048]
        self.is_causal = True
        self.expected_out_shape = [1, 2048, 8, 0]
        self.dtype = 'float16'


class TestZeroSizeCase5(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [2, 1, 8, 96]
        self.key_shape = [2, 100, 8, 96]
        self.value_shape = [2, 100, 8, 0]
        self.attn_mask_shape = None
        self.is_causal = False
        self.expected_out_shape = [2, 1, 8, 0]
        self.dtype = 'float16'


class TestZeroSizeCase6(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [2, 1, 8, 96]
        self.key_shape = [2, 101, 8, 96]
        self.value_shape = [2, 101, 8, 0]
        self.attn_mask_shape = None
        self.is_causal = False
        self.expected_out_shape = [2, 1, 8, 0]
        self.dtype = 'float16'


class TestZeroSizeCase7(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [1, 1024, 0, 128]
        self.key_shape = [1, 1024, 0, 128]
        self.value_shape = [1, 1024, 0, 128]
        self.attn_mask_shape = None
        self.is_causal = True
        self.expected_out_shape = [1, 1024, 0, 128]
        self.dtype = 'float16'


class TestZeroSizeCase8(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [1, 1024, 0, 16]
        self.key_shape = [1, 1024, 0, 16]
        self.value_shape = [1, 1024, 0, 16]
        self.attn_mask_shape = None
        self.is_causal = True
        self.expected_out_shape = [1, 1024, 0, 16]
        self.dtype = 'float16'


class TestZeroSizeCase9(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [1, 2048, 0, 16]
        self.key_shape = [1, 2048, 0, 16]
        self.value_shape = [1, 2048, 0, 16]
        self.attn_mask_shape = [1, 1, 0, 2048]
        self.is_causal = True
        self.expected_out_shape = [1, 2048, 0, 16]
        self.dtype = 'float16'


class TestZeroSizeCase10(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [1, 2048, 0, 64]
        self.key_shape = [1, 2048, 0, 64]
        self.value_shape = [1, 2048, 0, 64]
        self.attn_mask_shape = [1, 1, 0, 2048]
        self.is_causal = True
        self.expected_out_shape = [1, 2048, 0, 64]
        self.dtype = 'float16'


class TestZeroSizeCase11(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [2, 1, 0, 96]
        self.key_shape = [2, 100, 0, 96]
        self.value_shape = [2, 100, 0, 96]
        self.attn_mask_shape = None
        self.is_causal = False
        self.expected_out_shape = [2, 1, 0, 96]
        self.dtype = 'float16'


class TestZeroSizeCase12(TestZeroSizeBase):
    def setUp(self):
        self.query_shape = [2, 1, 0, 96]
        self.key_shape = [2, 101, 0, 96]
        self.value_shape = [2, 101, 0, 96]
        self.attn_mask_shape = None
        self.is_causal = False
        self.expected_out_shape = [2, 1, 0, 96]
        self.dtype = 'float16'


class TestZeroSizeCase13(TestZeroSizeBase):
    def setUp(self):
        # Case: Batch size is 0
        self.query_shape = [0, 32, 8, 64]
        self.key_shape = [0, 32, 8, 64]
        self.value_shape = [0, 32, 8, 64]
        self.attn_mask_shape = None
        self.is_causal = True
        self.expected_out_shape = [0, 32, 8, 64]
        self.dtype = 'float16'


class TestZeroSizeCase14(TestZeroSizeBase):
    def setUp(self):
        # Case: Batch size is 0, and NumHeads is 0
        self.query_shape = [0, 32, 0, 64]
        self.key_shape = [0, 32, 0, 64]
        self.value_shape = [0, 32, 0, 64]
        self.attn_mask_shape = None
        self.is_causal = True
        self.expected_out_shape = [0, 32, 0, 64]
        self.dtype = 'float16'


@unittest.skipIf(
    not mea_available,
    "Memory efficient attention is not available, skip TestMemEffAttnMaskBroadcasting",
)
class TestMemEffAttnMaskBroadcasting(unittest.TestCase):
    """
    Test case specifically for validating mask broadcasting logic in memory_efficient_attention.
    Target issue: Fix crash when mask shape is [B, 1, S, S] and Query is [B, H, S, D] where B > 1.
    """

    def setUp(self):
        self.place = get_device_place()
        self.dtype = 'float32'  # cutlass usually supports fp32/fp16
        self.dropout = 0.0
        self.causal = False

        # Key configuration to trigger the bug:
        # 1. Batch Size > 1
        # 2. Query Heads > 1
        # 3. Mask Heads == 1 (Requires broadcast)
        self.batch_size = 3
        self.seq_len = 8
        self.num_heads = 4
        self.head_dim = 16

        self.q_shape = (
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim,
        )
        # Mask shape: [Batch, 1, Seq, Seq]
        # This matches Query on Batch (3==3), but mismatch on Heads (1!=4)
        self.mask_shape = (self.batch_size, 1, self.seq_len, self.seq_len)

    def test_broadcast_mask_batch_match_head_broadcast(self):
        paddle.disable_static()

        # Create Inputs
        query = np.random.random(self.q_shape).astype(self.dtype)
        key = np.random.random(self.q_shape).astype(self.dtype)
        value = np.random.random(self.q_shape).astype(self.dtype)
        mask = np.random.random(self.mask_shape).astype(self.dtype)

        q = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        # Create Reference Inputs (Clone)
        q_ref = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k_ref = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v_ref = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m_ref = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        # 1. Run with Memory Efficient Attention (The one you fixed)
        # We enforce ONLY mem_efficient to make sure we hit the modified code path
        with sdp_kernel(
            enable_math=False, enable_flash=False, enable_mem_efficient=True
        ):
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=m,
                dropout_p=self.dropout,
                is_causal=self.causal,
            )

        # 2. Run with Naive Math Implementation (Reference)
        # Using the helper function defined in your file
        out_ref = attention_naive_with_mask(q_ref, k_ref, v_ref, m_ref)

        # 3. Validation
        # Check Output values
        np.testing.assert_allclose(
            out.numpy(), out_ref.numpy(), rtol=5e-3, atol=1e-3
        )

        # Check Gradients (Optional but good practice)
        out.backward()
        out_ref.backward()
        np.testing.assert_allclose(
            q.grad.numpy(), q_ref.grad.numpy(), rtol=5e-3, atol=1e-3
        )

    def test_broadcast_mask_head_match_batch_broadcast(self):
        paddle.disable_static()

        query = np.random.random(self.q_shape).astype(self.dtype)
        key = np.random.random(self.q_shape).astype(self.dtype)
        value = np.random.random(self.q_shape).astype(self.dtype)
        mask = np.random.random(
            [1, self.num_heads, self.seq_len, self.seq_len]
        ).astype(self.dtype)

        q = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        q_ref = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k_ref = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v_ref = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m_ref = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        with sdp_kernel(
            enable_math=False, enable_flash=False, enable_mem_efficient=True
        ):
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=m,
                dropout_p=self.dropout,
                is_causal=self.causal,
            )

        out_ref = attention_naive_with_mask(q_ref, k_ref, v_ref, m_ref)

        np.testing.assert_allclose(
            out.numpy(), out_ref.numpy(), rtol=5e-3, atol=1e-3
        )

        out.backward()
        out_ref.backward()
        np.testing.assert_allclose(
            q.grad.numpy(), q_ref.grad.numpy(), rtol=5e-3, atol=1e-3
        )

    def test_broadcast_mask_double_broadcast(self):
        """
        Test extreme case: [1, 1, S, S] -> [B, H, S, S]
        This was technically supported before, but good to regression test.
        """
        paddle.disable_static()

        # Mask shape: [1, 1, Seq, Seq]
        broadcast_mask_shape = (1, 1, self.seq_len, self.seq_len)

        query = np.random.random(self.q_shape).astype(self.dtype)
        mask = np.random.random(broadcast_mask_shape).astype(self.dtype)

        q = paddle.to_tensor(query, place=self.place)
        k = paddle.to_tensor(query, place=self.place)  # shared for simplicity
        v = paddle.to_tensor(query, place=self.place)
        m = paddle.to_tensor(mask, place=self.place)

        # Should pass without error
        with sdp_kernel(
            enable_math=False, enable_flash=False, enable_mem_efficient=True
        ):
            out = scaled_dot_product_attention(
                q, k, v, attn_mask=m, dropout_p=self.dropout
            )
        self.assertEqual(out.shape, list(self.q_shape))

    def _run_test_with_shape(self, mask_shape, case_name):
        """
        Helper function to run test with a specific mask shape.
        Handles data creation, running both MEA and Naive implementations, and validation.
        """
        paddle.disable_static()

        # 1. Prepare Inputs
        query = np.random.random(self.q_shape).astype(self.dtype)
        key = np.random.random(self.q_shape).astype(self.dtype)
        value = np.random.random(self.q_shape).astype(self.dtype)
        mask = np.random.random(mask_shape).astype(self.dtype)

        q = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        q_ref = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k_ref = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v_ref = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m_ref = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        # 2. Run Memory Efficient Attention (Target)
        with sdp_kernel(
            enable_math=False, enable_flash=False, enable_mem_efficient=True
        ):
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=m,
                is_causal=self.causal,
            )

        # 3. Run Naive Math (Reference)
        # Note: Standard broadcasting rules apply for the reference implementation
        # 2D [S, S] broadcasts to [B, H, S, S]
        # 3D [B, S, S] broadcasts to [B, H, S, S]
        out_ref = attention_naive_with_mask(q_ref, k_ref, v_ref, m_ref)

        # 4. Validation
        np.testing.assert_allclose(
            out.numpy(),
            out_ref.numpy(),
            rtol=5e-3,
            atol=1e-3,
            err_msg=f"Output mismatch in case: {case_name}",
        )

        out.backward()
        out_ref.backward()
        np.testing.assert_allclose(
            q.grad.numpy(),
            q_ref.grad.numpy(),
            rtol=5e-3,
            atol=1e-3,
            err_msg=f"Gradient mismatch in case: {case_name}",
        )

    def test_2d_mask(self):
        """
        Case: 2D Mask [Seq, Seq]
        Expectation: Automatically broadcasts to [B, H, Seq, Seq]
        """
        mask_shape = (self.seq_len, self.seq_len)
        self._run_test_with_shape(mask_shape, "test_2d_mask")

    def test_3d_mask_batch_match(self):
        """
        Case: 3D Mask [B, Seq, Seq] where B > 1 matches Query
        Expectation: Broadcasts to [B, H, Seq, Seq]
        """
        mask_shape = (self.batch_size, self.seq_len, self.seq_len)
        self._run_test_with_shape(mask_shape, "test_3d_mask_batch_match")

    def test_3d_mask_batch_broadcast(self):
        """
        Case: 3D Mask [1, Seq, Seq]
        Expectation: Broadcasts to [B, H, Seq, Seq]
        """
        mask_shape = (1, self.seq_len, self.seq_len)
        self._run_test_with_shape(mask_shape, "test_3d_mask_batch_broadcast")

    def test_4d_mask_full_match(self):
        """
        Case: 4D Mask [B, H, Seq, Seq] (Full Match)
        Expectation: Works natively without broadcasting
        """
        mask_shape = (
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.seq_len,
        )
        self._run_test_with_shape(mask_shape, "test_4d_mask_full_match")


@unittest.skipIf(
    not fa_available,
    "Flash attention is not available, skip TestFlashAttnMaskLogic",
)
class TestFlashAttnMaskLogic(unittest.TestCase):
    """
    Test case specifically for validating mask unsqueeze logic in flash_attention path.
    Target logic:
      - 2D Mask [S, S] -> Unsqueeze to [1, 1, S, S]
      - 3D Mask [B, S, S] -> Unsqueeze to [B, 1, S, S]
    """

    def setUp(self):
        self.place = get_device_place()
        # FlashAttn usually requires fp16 or bf16 on most cards.
        # Using float16 to ensure we hit the FA kernel logic.
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False

        self.batch_size = 2
        self.seq_len = 128
        self.num_heads = 4
        self.head_dim = 32

        self.q_shape = (
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim,
        )
        # Note: K/V shape is same as Q here for simplicity

    def _run_test_with_mask_shape(self, mask_shape):
        paddle.disable_static()

        # 1. Create Inputs in FP16
        query = np.random.random(self.q_shape).astype("float16")
        key = np.random.random(self.q_shape).astype("float16")
        value = np.random.random(self.q_shape).astype("float16")
        mask = np.random.random(mask_shape).astype("float16")

        q = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        # 2. Create Reference Inputs (Cast to FP32 for higher precision reference)
        q_ref = paddle.cast(q, 'float32')
        k_ref = paddle.cast(k, 'float32')
        v_ref = paddle.cast(v, 'float32')
        m_ref = paddle.cast(m, 'float32')
        if m_ref.ndim == 2:
            m_ref = m_ref.unsqueeze([0, 1])
        elif m_ref.ndim == 3:
            m_ref = m_ref.unsqueeze(1)

        # 3. Run with Flash Attention (Force Enable)
        # This will trigger your Python logic: if ndim==2/3 -> unsqueeze -> call _C_ops.flash_attn
        try:
            with sdp_kernel(
                enable_math=False, enable_flash=True, enable_mem_efficient=False
            ):
                out = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=m,
                    dropout_p=self.dropout,
                    is_causal=self.causal,
                )
        except RuntimeError as e:
            # Skip if hardware doesn't support FA (e.g. old GPU or dim mismatch restrictions)
            if "No available backend" in str(e) or "not support" in str(e):
                print(
                    f"Skipping FlashAttn test for shape {mask_shape} due to hardware/library constraints."
                )
                return
            raise e

        # 4. Run with Naive Math Implementation (Reference)
        # Paddle's basic math ops handle broadcasting automatically:
        # [B, H, S, S] + [S, S] -> Works
        # [B, H, S, S] + [B, S, S] -> Works
        out_ref = attention_naive_with_mask(q_ref, k_ref, v_ref, m_ref)

        # 5. Validation
        # Cast FA output back to FP32 for comparison
        out_fp32 = paddle.cast(out, 'float32')

        # Relax tolerance slightly for FP16 vs FP32 comparison
        np.testing.assert_allclose(
            out_fp32.numpy(), out_ref.numpy(), rtol=1e-2, atol=1e-2
        )

    def test_flash_attn_2d_mask(self):
        """
        Test passing a [S, S] mask to Flash Attention.
        Should internally unsqueeze to [1, 1, S, S].
        """
        # Shape: [Seq, Seq]
        mask_shape = (self.seq_len, self.seq_len)
        self._run_test_with_mask_shape(mask_shape)

    def test_flash_attn_3d_mask(self):
        """
        Test passing a [B, S, S] mask to Flash Attention.
        Should internally unsqueeze to [B, 1, S, S].
        """
        # Shape: [Batch, Seq, Seq]
        mask_shape = (self.batch_size, self.seq_len, self.seq_len)
        self._run_test_with_mask_shape(mask_shape)


@unittest.skipIf(
    paddle.device.is_compiled_with_xpu(),
    "SDPA on XPU force select FA backend, skip math broadcast test.",
)
class TestMathAttnMaskBroadcasting(unittest.TestCase):
    """
    Test case specifically for validating mask broadcasting logic in math_attention.
    Target issue: Fix crash when mask shape is [B, 1, S, S] and Query is [B, H, S, D] where B > 1.
    """

    def setUp(self):
        self.place = get_device_place()
        self.dtype = 'float32'
        self.dropout = 0.0
        self.causal = False

        # Key configuration to trigger the bug:
        # 1. Batch Size > 1
        # 2. Query Heads > 1
        # 3. Mask Heads == 1 (Requires broadcast)
        self.batch_size = 3
        self.seq_len = 8
        self.num_heads = 4
        self.head_dim = 16

        self.q_shape = (
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim,
        )
        # Mask shape: [Batch, 1, Seq, Seq]
        # This matches Query on Batch (3==3), but mismatch on Heads (1!=4)
        self.mask_shape = (self.batch_size, 1, self.seq_len, self.seq_len)

    def test_broadcast_mask_batch_match_head_broadcast(self):
        paddle.disable_static()

        # Create Inputs
        query = np.random.random(self.q_shape).astype(self.dtype)
        key = np.random.random(self.q_shape).astype(self.dtype)
        value = np.random.random(self.q_shape).astype(self.dtype)
        mask = np.random.random(self.mask_shape).astype(self.dtype)

        q = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        # Create Reference Inputs (Clone)
        q_ref = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k_ref = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v_ref = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m_ref = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        with sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False
        ):
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=m,
                dropout_p=self.dropout,
                is_causal=self.causal,
            )

        # 2. Run with Naive Math Implementation (Reference)
        # Using the helper function defined in your file
        out_ref = attention_naive_with_mask(q_ref, k_ref, v_ref, m_ref)

        # 3. Validation
        # Check Output values
        np.testing.assert_allclose(
            out.numpy(), out_ref.numpy(), rtol=5e-3, atol=1e-3
        )

        # Check Gradients (Optional but good practice)
        out.backward()
        out_ref.backward()
        np.testing.assert_allclose(
            q.grad.numpy(), q_ref.grad.numpy(), rtol=5e-3, atol=1e-3
        )

    def test_broadcast_mask_head_match_batch_broadcast(self):
        paddle.disable_static()

        query = np.random.random(self.q_shape).astype(self.dtype)
        key = np.random.random(self.q_shape).astype(self.dtype)
        value = np.random.random(self.q_shape).astype(self.dtype)
        mask = np.random.random(
            [1, self.num_heads, self.seq_len, self.seq_len]
        ).astype(self.dtype)

        q = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        q_ref = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k_ref = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v_ref = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m_ref = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        with sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False
        ):
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=m,
                dropout_p=self.dropout,
                is_causal=self.causal,
            )

        out_ref = attention_naive_with_mask(q_ref, k_ref, v_ref, m_ref)

        np.testing.assert_allclose(
            out.numpy(), out_ref.numpy(), rtol=5e-3, atol=1e-3
        )

        out.backward()
        out_ref.backward()
        np.testing.assert_allclose(
            q.grad.numpy(), q_ref.grad.numpy(), rtol=5e-3, atol=1e-3
        )

    def test_broadcast_mask_double_broadcast(self):
        """
        Test extreme case: [1, 1, S, S] -> [B, H, S, S]
        This was technically supported before, but good to regression test.
        """
        paddle.disable_static()

        # Mask shape: [1, 1, Seq, Seq]
        broadcast_mask_shape = (1, 1, self.seq_len, self.seq_len)

        query = np.random.random(self.q_shape).astype(self.dtype)
        mask = np.random.random(broadcast_mask_shape).astype(self.dtype)

        q = paddle.to_tensor(query, place=self.place)
        k = paddle.to_tensor(query, place=self.place)  # shared for simplicity
        v = paddle.to_tensor(query, place=self.place)
        m = paddle.to_tensor(mask, place=self.place)

        # Should pass without error
        with sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False
        ):
            out = scaled_dot_product_attention(
                q, k, v, attn_mask=m, dropout_p=self.dropout
            )
        self.assertEqual(out.shape, list(self.q_shape))

    def _run_test_with_shape(self, mask_shape, case_name):
        """
        Helper function to run test with a specific mask shape.
        Handles data creation, running both MEA and Naive implementations, and validation.
        """
        paddle.disable_static()

        # 1. Prepare Inputs
        query = np.random.random(self.q_shape).astype(self.dtype)
        key = np.random.random(self.q_shape).astype(self.dtype)
        value = np.random.random(self.q_shape).astype(self.dtype)
        mask = np.random.random(mask_shape).astype(self.dtype)

        q = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        q_ref = paddle.to_tensor(query, place=self.place, stop_gradient=False)
        k_ref = paddle.to_tensor(key, place=self.place, stop_gradient=False)
        v_ref = paddle.to_tensor(value, place=self.place, stop_gradient=False)
        m_ref = paddle.to_tensor(mask, place=self.place, stop_gradient=False)

        with sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False
        ):
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=m,
                is_causal=self.causal,
            )

        # 3. Run Naive Math (Reference)
        # Note: Standard broadcasting rules apply for the reference implementation
        # 2D [S, S] broadcasts to [B, H, S, S]
        # 3D [B, S, S] broadcasts to [B, H, S, S]
        out_ref = attention_naive_with_mask(q_ref, k_ref, v_ref, m_ref)

        # 4. Validation
        np.testing.assert_allclose(
            out.numpy(),
            out_ref.numpy(),
            rtol=5e-3,
            atol=1e-3,
            err_msg=f"Output mismatch in case: {case_name}",
        )

        out.backward()
        out_ref.backward()
        np.testing.assert_allclose(
            q.grad.numpy(),
            q_ref.grad.numpy(),
            rtol=5e-3,
            atol=1e-3,
            err_msg=f"Gradient mismatch in case: {case_name}",
        )

    def test_2d_mask(self):
        """
        Case: 2D Mask [Seq, Seq]
        Expectation: Automatically broadcasts to [B, H, Seq, Seq]
        """
        mask_shape = (self.seq_len, self.seq_len)
        self._run_test_with_shape(mask_shape, "test_2d_mask")

    def test_3d_mask_batch_match(self):
        """
        Case: 3D Mask [B, Seq, Seq] where B > 1 matches Query
        Expectation: Broadcasts to [B, H, Seq, Seq]
        """
        mask_shape = (self.batch_size, self.seq_len, self.seq_len)
        self._run_test_with_shape(mask_shape, "test_3d_mask_batch_match")

    def test_3d_mask_batch_broadcast(self):
        """
        Case: 3D Mask [1, Seq, Seq]
        Expectation: Broadcasts to [B, H, Seq, Seq]
        """
        mask_shape = (1, self.seq_len, self.seq_len)
        self._run_test_with_shape(mask_shape, "test_3d_mask_batch_broadcast")

    def test_4d_mask_full_match(self):
        """
        Case: 4D Mask [B, H, Seq, Seq] (Full Match)
        Expectation: Works natively without broadcasting
        """
        mask_shape = (
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.seq_len,
        )
        self._run_test_with_shape(mask_shape, "test_4d_mask_full_match")


if __name__ == '__main__':
    unittest.main()
