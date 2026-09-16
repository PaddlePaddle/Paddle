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
"""
Test that paddle.nn.LayerNorm correctly handles bfloat16 input on HIP/ROCm.

Root cause fixed:
  paddle/phi/kernels/gpu/layer_norm_kernel.cu: HIP PD_REGISTER_KERNEL did not
  include phi::bfloat16. This test verifies the fix is present and working.
"""

import unittest

import numpy as np

import paddle
import paddle.nn as nn


def _ref_layer_norm_fp32(x_np, weight_np, bias_np, eps=1e-5):
    """NumPy reference: layer norm over last dimension."""
    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    y = (x_np - mean) / np.sqrt(var + eps)
    if weight_np is not None:
        y = y * weight_np
    if bias_np is not None:
        y = y + bias_np
    return y


@unittest.skipUnless(
    paddle.is_compiled_with_rocm() or paddle.is_compiled_with_cuda(),
    "GPU test: requires CUDA or ROCm",
)
class TestLayerNormBF16HIP(unittest.TestCase):
    """Verify LayerNorm kernel is registered for bfloat16 on HIP."""

    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        np.random.seed(42)

    def _run_layer_norm_bf16(self, shape, normalized_shape):
        x_fp32 = np.random.randn(*shape).astype(np.float32)
        weight_fp32 = np.ones(normalized_shape, dtype=np.float32)
        bias_fp32 = np.zeros(normalized_shape, dtype=np.float32)

        # Reference: compute in fp32
        ref = _ref_layer_norm_fp32(x_fp32, weight_fp32, bias_fp32)

        # BF16 path
        x_bf16 = paddle.to_tensor(
            x_fp32, dtype=paddle.bfloat16, place=self.place
        )
        ln = nn.LayerNorm(normalized_shape)
        ln = ln.to(dtype=paddle.bfloat16)

        with paddle.amp.auto_cast(dtype='bfloat16'):
            out_bf16 = ln(x_bf16)

        out_fp32 = out_bf16.cast(paddle.float32).numpy()

        # BF16 has ~2 decimal digits of precision; use generous tolerance
        np.testing.assert_allclose(
            out_fp32, ref, rtol=1e-2, atol=1e-2,
            err_msg=f"LayerNorm BF16 output does not match FP32 reference "
                    f"(shape={shape}, normalized_shape={normalized_shape})",
        )

    def test_2d_last_dim(self):
        """LayerNorm over last dim of a 2D tensor."""
        self._run_layer_norm_bf16((4, 64), (64,))

    def test_3d_last_dim(self):
        """LayerNorm over last dim of a 3D tensor (batch, seq, hidden)."""
        self._run_layer_norm_bf16((2, 8, 128), (128,))

    def test_4d_last_two_dims(self):
        """LayerNorm over last two dims of a 4D tensor."""
        self._run_layer_norm_bf16((2, 4, 16, 16), (16, 16))

    def test_bf16_output_dtype(self):
        """Output dtype must be bfloat16 when input is bfloat16."""
        x = paddle.ones([2, 32], dtype=paddle.bfloat16)
        ln = nn.LayerNorm(32).to(dtype=paddle.bfloat16)
        out = ln(x)
        self.assertEqual(
            out.dtype,
            paddle.bfloat16,
            "LayerNorm output dtype should be bfloat16 when input is bfloat16",
        )

    def test_kernel_registered(self):
        """Verify that calling layer_norm with bfloat16 does not raise
        'kernel not registered' error (the core fix for HIP)."""
        x = paddle.randn([4, 64]).cast(paddle.bfloat16)
        try:
            out = paddle.nn.functional.layer_norm(x, [64])
        except Exception as e:
            self.fail(
                f"layer_norm raised an exception with bfloat16 input: {e}"
            )

    @unittest.skipUnless(
        paddle.is_compiled_with_rocm(),
        "ROCm-specific test",
    )
    def test_rocm_bf16_snr(self):
        """On ROCm, verify Signal-to-Noise ratio >= 30 dB vs FP32 reference."""
        np.random.seed(0)
        x_fp32_np = np.random.randn(8, 256).astype(np.float32)

        x_fp32 = paddle.to_tensor(x_fp32_np, place=self.place)
        ln_fp32 = nn.LayerNorm(256)
        ref_out = ln_fp32(x_fp32).numpy()

        x_bf16 = x_fp32.cast(paddle.bfloat16)
        ln_bf16 = nn.LayerNorm(256).to(dtype=paddle.bfloat16)
        test_out = ln_bf16(x_bf16).cast(paddle.float32).numpy()

        signal_power = np.mean(ref_out ** 2)
        noise_power = np.mean((ref_out - test_out) ** 2) + 1e-30
        snr_db = 10 * np.log10(signal_power / noise_power)

        self.assertGreaterEqual(
            snr_db, 30.0,
            f"LayerNorm BF16 SNR on ROCm too low: {snr_db:.1f} dB (expected >= 30 dB)",
        )
        print(f"\n[ROCm BF16 LayerNorm SNR] {snr_db:.1f} dB — PASS")


if __name__ == '__main__':
    unittest.main()
