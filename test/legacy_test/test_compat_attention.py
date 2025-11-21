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

import math
import unittest

import numpy as np
from op_test import get_cuda_version, get_device_place, is_custom_device

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.compat.nn.functional import (
    scaled_dot_product_attention as compat_sdpa,
)
from paddle.nn.functional import (
    scaled_dot_product_attention as legacy_sdpa,
)

is_sm8x = (
    (core.is_compiled_with_cuda() or is_custom_device())
    and paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] >= 0
)
is_sm90 = (
    (core.is_compiled_with_cuda() or is_custom_device())
    and paddle.device.cuda.get_device_capability()[0] == 9
    and paddle.device.cuda.get_device_capability()[1] == 0
)
is_sm_supported = is_sm8x or is_sm90


def is_flashattn_supported():
    if (
        not (core.is_compiled_with_cuda() or is_custom_device())
        or get_cuda_version() < 11040
        or not is_sm_supported
    ):
        return False
    return True


def _attention_naive_bnsd(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    q_ndim = query.ndim
    L, S = query.shape[-2], key.shape[-2]
    E = query.shape[-1]

    scale_factor = scale if scale is not None else (1.0 / math.sqrt(E))

    attn_bias = None
    if is_causal:
        if attn_mask is not None:
            raise ValueError("Cannot set both attn_mask and is_causal=True")
        temp_mask = paddle.ones([L, S], dtype='bool').tril(diagonal=0)
        attn_bias = paddle.where(
            temp_mask.logical_not(),
            paddle.to_tensor(-float('inf'), dtype=query.dtype),
            paddle.to_tensor(0.0, dtype=query.dtype),
        )

    if attn_mask is not None:
        if attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)
        # (L, S) -> (1, 1, L, S)
        elif attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        if attn_mask.dtype == paddle.bool:
            attn_bias = paddle.where(
                attn_mask.logical_not(),
                paddle.to_tensor(-float('inf'), dtype=query.dtype),
                paddle.to_tensor(0.0, dtype=query.dtype),
            )
        else:
            attn_bias = attn_mask

    if enable_gqa:
        q_heads, k_heads, v_heads = (
            query.shape[-3],
            key.shape[-3],
            value.shape[-3],
        )
        if q_heads % k_heads != 0:
            raise ValueError("GQA heads mismatch K")
        if q_heads % v_heads != 0:
            raise ValueError("GQA heads mismatch V")

        if k_heads != q_heads:
            repeats = q_heads // k_heads
            key = (
                key.unsqueeze(-3)
                .expand([-1] * (key.ndim - 2) + [repeats, -1, -1])
                .flatten(start_axis=-4, stop_axis=-3)
            )
        if v_heads != q_heads:
            repeats = q_heads // v_heads
            value = (
                value.unsqueeze(-3)
                .expand([-1] * (value.ndim - 2) + [repeats, -1, -1])
                .flatten(start_axis=-4, stop_axis=-3)
            )

    perm = list(range(key.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    key_transposed = paddle.transpose(key, perm)
    attn_weight = paddle.matmul(query, key_transposed) * scale_factor

    if attn_bias is not None:
        attn_weight = attn_weight + attn_bias

    attn_weight = F.softmax(attn_weight, axis=-1)

    if dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=dropout_p, training=True)

    return paddle.matmul(attn_weight, value)


@unittest.skipIf(
    not is_flashattn_supported(),
    "compat.sdpa test requires CUDA 11.4+ and SM 8.0+",
)
class TestCompatSDPAFP16(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()
        self.set_dtype()
        self.rtol = 1e-2
        self.atol = 1e-2

        self.B = 2
        self.S = 128  # Seq Len
        self.H = 8  # Num Heads
        self.D = 16  # Head Dim

        self.shape_bnsd = (self.B, self.H, self.S, self.D)
        self.shape_bshd = (self.B, self.S, self.H, self.D)

        self.H_GQA_K = 2
        self.shape_bnsd_q = (self.B, self.H, self.S, self.D)
        self.shape_bnsd_k = (self.B, self.H_GQA_K, self.S, self.D)
        self.shape_bnsd_v = (self.B, self.H_GQA_K, self.S, self.D)

        paddle.disable_static()

    def set_dtype(self):
        self.dtype = paddle.float16

    def _transpose_bnsd_to_bshd(self, x):
        return x.transpose([0, 2, 1, 3])

    def _transpose_bshd_to_bnsd(self, x):
        return x.transpose([0, 2, 1, 3])

    def test_fast_path_bnsd_vs_legacy_bshd(self):
        q_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        k_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        v_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)

        out_compat = compat_sdpa(
            q_bnsd, k_bnsd, v_bnsd, scale=None, enable_gqa=False
        )

        q_bshd = self._transpose_bnsd_to_bshd(q_bnsd)
        k_bshd = self._transpose_bnsd_to_bshd(k_bnsd)
        v_bshd = self._transpose_bnsd_to_bshd(v_bnsd)

        out_legacy_raw = legacy_sdpa(
            q_bshd, k_bshd, v_bshd, scale=None, training=True
        )

        out_legacy = self._transpose_bshd_to_bnsd(out_legacy_raw)

        self.assertEqual(out_compat.shape, list(self.shape_bnsd))
        self.assertEqual(out_legacy.shape, list(self.shape_bnsd))
        np.testing.assert_allclose(
            out_compat.numpy(),
            out_legacy.numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_compat_path_gqa_vs_naive(self):
        q_bnsd = paddle.randn(self.shape_bnsd_q, dtype=self.dtype)
        k_bnsd_gqa = paddle.randn(self.shape_bnsd_k, dtype=self.dtype)
        v_bnsd_gqa = paddle.randn(self.shape_bnsd_v, dtype=self.dtype)

        out_compat = compat_sdpa(
            q_bnsd, k_bnsd_gqa, v_bnsd_gqa, scale=None, enable_gqa=True
        )

        out_naive = _attention_naive_bnsd(
            q_bnsd, k_bnsd_gqa, v_bnsd_gqa, scale=None, enable_gqa=True
        )

        self.assertEqual(out_compat.shape, list(self.shape_bnsd_q))
        np.testing.assert_allclose(
            out_compat.numpy(),
            out_naive.numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_compat_path_scale_vs_naive(self):
        custom_scale = 0.5
        q_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        k_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        v_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)

        out_compat = compat_sdpa(
            q_bnsd, k_bnsd, v_bnsd, scale=custom_scale, enable_gqa=False
        )

        out_naive = _attention_naive_bnsd(
            q_bnsd, k_bnsd, v_bnsd, scale=custom_scale, enable_gqa=False
        )

        self.assertEqual(out_compat.shape, list(self.shape_bnsd))
        np.testing.assert_allclose(
            out_compat.numpy(),
            out_naive.numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_compat_path_dropout(self):
        q_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        k_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        v_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)

        paddle.seed(42)
        out1 = compat_sdpa(
            q_bnsd, k_bnsd, v_bnsd, dropout_p=0.1, enable_gqa=False
        )
        paddle.seed(43)
        out2 = compat_sdpa(
            q_bnsd, k_bnsd, v_bnsd, dropout_p=0.1, enable_gqa=False
        )

        self.assertFalse(
            np.array_equal(out1.numpy(), out2.numpy()),
            "Dropout p>0, outputs should be different",
        )

        paddle.seed(42)
        out3 = compat_sdpa(
            q_bnsd, k_bnsd, v_bnsd, dropout_p=0.0, enable_gqa=False
        )
        paddle.seed(43)
        out4 = compat_sdpa(
            q_bnsd, k_bnsd, v_bnsd, dropout_p=0.0, enable_gqa=False
        )
        np.testing.assert_allclose(
            out3.numpy(),
            out4.numpy(),
            err_msg="Dropout p=0, outputs should be identical",
        )

    def test_compat_path_mask_3d_broadcast(self):
        q_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        k_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        v_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)

        mask_3d = paddle.randn([self.B, self.S, self.S], dtype=self.dtype)

        out_compat = compat_sdpa(
            q_bnsd, k_bnsd, v_bnsd, attn_mask=mask_3d, scale=None
        )

        out_naive = _attention_naive_bnsd(
            q_bnsd, k_bnsd, v_bnsd, attn_mask=mask_3d, scale=None
        )

        self.assertEqual(out_compat.shape, list(self.shape_bnsd))
        np.testing.assert_allclose(
            out_compat.numpy(),
            out_naive.numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_compat_path_mask_2d_broadcast(self):
        q_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        k_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)
        v_bnsd = paddle.randn(self.shape_bnsd, dtype=self.dtype)

        mask_2d = paddle.randn([self.S, self.S], dtype=self.dtype)

        out_compat = compat_sdpa(
            q_bnsd, k_bnsd, v_bnsd, attn_mask=mask_2d, scale=None
        )

        out_naive = _attention_naive_bnsd(
            q_bnsd, k_bnsd, v_bnsd, attn_mask=mask_2d, scale=None
        )

        self.assertEqual(out_compat.shape, list(self.shape_bnsd))
        np.testing.assert_allclose(
            out_compat.numpy(),
            out_naive.numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )


class TestCompatSDPABF16(unittest.TestCase):
    def set_dtype(self):
        return paddle.bfloat16


if __name__ == '__main__':
    unittest.main()
