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

import unittest

import numpy as np

import paddle


def _has_sm80():
    if not paddle.is_compiled_with_cuda():
        return False
    cap = paddle.device.cuda.get_device_capability()
    return cap is not None and cap[0] >= 8


_skip = "fused_swiglu_weighted_bwd_clamped requires GPU SM80+ with bfloat16"
FUNC = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd_clamped
FUNC_OLD = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd


def _make_data(seq_len=256, topk=2, mid=128):
    o1 = paddle.rand([topk, seq_len, mid * 2], dtype="bfloat16")
    probs = paddle.rand([topk, seq_len, 1], dtype="float32")
    do2_s = paddle.rand([topk, seq_len, mid], dtype="bfloat16")
    return o1, probs, do2_s


def _gold_clamped_bwd(o1, unzipped_probs, do2_s, cv):
    """Python reference for clamped weighted SwiGLU backward."""
    h = o1.shape[-1] // 2
    o1_f = o1.cast("float32")
    g_raw, v_raw = o1_f[..., :h], o1_f[..., h:]
    gate = paddle.clip(g_raw, max=cv)
    val = paddle.clip(v_raw, min=-cv, max=cv)
    g_mask = (g_raw <= cv).cast("float32")
    v_mask = ((v_raw <= cv) & (v_raw >= -cv)).cast("float32")
    sig = paddle.nn.functional.sigmoid(gate)
    silu = gate * sig
    swiglu_val = silu * val
    o2_s = (swiglu_val * unzipped_probs).cast("bfloat16")
    do2 = do2_s.cast("float32") * unzipped_probs
    d_gate = do2 * val * sig * (1.0 + gate * (1.0 - sig)) * g_mask
    d_val = do2 * silu * v_mask
    do1 = paddle.concat([d_gate, d_val], axis=-1).cast("bfloat16")
    probs_grad = (do2_s.cast("float32") * swiglu_val).sum(axis=-1, keepdim=True)
    return do1, probs_grad, o2_s


def _assert_close(a, b, rtol=2e-2, atol=2e-2):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@unittest.skipUnless(_has_sm80(), _skip)
class TestFusedWeightedSwigluBwdClamp(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)

    def test_large_clamp_matches_no_clamp(self):
        """Large clamp_value (1e9) matches the no-clamp op exactly."""
        o1, probs, do2_s = _make_data()
        do1_old, pg_old, o2s_old = FUNC_OLD(o1, do2_s, probs)
        do1_new, pg_new, o2s_new = FUNC(o1, do2_s, probs, clamp_value=1e9)
        _assert_close(
            do1_old.cast("float32").numpy(),
            do1_new.cast("float32").numpy(),
            0,
            0,
        )
        _assert_close(pg_old.numpy(), pg_new.numpy(), 0, 0)
        _assert_close(
            o2s_old.cast("float32").numpy(),
            o2s_new.cast("float32").numpy(),
            0,
            0,
        )

    def test_clamp_saturates_gradients(self):
        """All inputs saturated → gradients zeroed."""
        mid = 128
        o1 = paddle.concat(
            [paddle.full([2, 64, mid], 5.0), paddle.full([2, 64, mid], 5.0)],
            axis=-1,
        ).cast("bfloat16")
        probs = paddle.ones([2, 64, 1], dtype="float32")
        do2_s = paddle.ones([2, 64, mid], dtype="bfloat16")
        do1, _, _ = FUNC(o1, do2_s, probs, clamp_value=1.0)
        np.testing.assert_allclose(do1.cast("float32").numpy(), 0.0, atol=1e-2)

    def test_clamp_matches_python_reference(self):
        """Fused kernel matches Python gold reference (Vec4 path)."""
        o1, probs, do2_s = _make_data(seq_len=128, topk=2, mid=256)
        do1_f, pg_f, o2s_f = FUNC(o1, do2_s, probs, clamp_value=6.0)
        do1_g, pg_g, o2s_g = _gold_clamped_bwd(o1, probs, do2_s, 6.0)
        _assert_close(
            do1_f.cast("float32").numpy(), do1_g.cast("float32").numpy()
        )
        _assert_close(pg_f.numpy(), pg_g.numpy())
        _assert_close(
            o2s_f.cast("float32").numpy(), o2s_g.cast("float32").numpy()
        )

    def test_clamp_partial_saturation(self):
        """Mix of saturated and non-saturated inputs."""
        mid, cv = 64, 2.0
        half = mid // 2
        sat = paddle.full([1, 1, half], 5.0)
        act = paddle.full([1, 1, half], 0.5)
        o1 = paddle.concat([sat, act, sat, act], axis=-1).cast("bfloat16")
        probs = paddle.ones([1, 1, 1], dtype="float32")
        do2_s = paddle.ones([1, 1, mid], dtype="bfloat16")
        do1, _, _ = FUNC(o1, do2_s, probs, clamp_value=cv)
        do1_f = do1.cast("float32").numpy()
        np.testing.assert_allclose(do1_f[0, 0, :half], 0.0, atol=1e-2)
        self.assertTrue(np.any(np.abs(do1_f[0, 0, half:mid]) > 1e-3))

    def test_clamp_output_shapes_and_dtypes(self):
        """clamp_value must not change output shapes or dtypes."""
        o1, probs, do2_s = _make_data()
        do1, pg, o2_s = FUNC(o1, do2_s, probs, clamp_value=5.0)
        self.assertEqual(list(do1.shape), list(o1.shape))
        self.assertEqual(list(pg.shape), list(probs.shape))
        self.assertEqual(
            list(o2_s.shape), [o1.shape[0], o1.shape[1], o1.shape[2] // 2]
        )
        self.assertEqual(do1.dtype, paddle.bfloat16)
        self.assertEqual(pg.dtype, paddle.float32)
        self.assertEqual(o2_s.dtype, paddle.bfloat16)

    def test_tiny_and_zero_clamp_value(self):
        """Tiny clamp_value bounds output; zero clamp_value zeros output."""
        mid = 64
        probs = paddle.ones([1, 32, 1], dtype="float32")
        do2_s = paddle.ones([1, 32, mid], dtype="bfloat16")
        # Tiny clamp
        o1 = paddle.randn([1, 32, mid * 2]).cast("bfloat16")
        _, _, o2_s = FUNC(o1, do2_s, probs, clamp_value=0.1)
        bound = (
            float(paddle.nn.functional.sigmoid(paddle.to_tensor(0.1)).numpy())
            * 0.1
        )
        self.assertLessEqual(
            float(o2_s.cast("float32").abs().max().numpy()), bound + 0.05
        )
        # Zero clamp: silu(0)*0 = 0
        o1 = paddle.randn([1, 32, mid * 2]).cast("bfloat16")
        do2_s2 = paddle.ones([1, 32, mid], dtype="bfloat16")
        _, _, o2_s = FUNC(o1, do2_s2, probs, clamp_value=0.0)
        self.assertLessEqual(
            float(o2_s.cast("float32").abs().max().numpy()), 1e-2
        )

    def test_negative_val_clamped(self):
        """val < -clamp_value clamped and gradients masked."""
        mid = 64
        o1 = paddle.concat(
            [paddle.full([1, 16, mid], 0.5), paddle.full([1, 16, mid], -5.0)],
            axis=-1,
        ).cast("bfloat16")
        probs = paddle.ones([1, 16, 1], dtype="float32")
        do2_s = paddle.ones([1, 16, mid], dtype="bfloat16")
        do1, _, _ = FUNC(o1, do2_s, probs, clamp_value=1.0)
        np.testing.assert_allclose(
            do1.cast("float32").numpy()[0, 0, mid:], 0.0, atol=1e-2
        )

    def test_clamp_reproducibility(self):
        """Results should be deterministic."""
        o1, probs, do2_s = _make_data()
        do1_a, pg_a, _ = FUNC(o1, do2_s, probs, clamp_value=3.0)
        do1_b, pg_b, _ = FUNC(o1, do2_s, probs, clamp_value=3.0)
        _assert_close(
            do1_a.cast("float32").numpy(), do1_b.cast("float32").numpy(), 0, 0
        )
        _assert_close(pg_a.numpy(), pg_b.numpy(), 0, 0)

    def test_scalar_kernel_path(self):
        """moe_intermediate_size not divisible by 4 → scalar kernel path."""
        mid = 66
        o1 = paddle.rand([2, 32, mid * 2], dtype="bfloat16")
        probs = paddle.rand([2, 32, 1], dtype="float32")
        do2_s = paddle.rand([2, 32, mid], dtype="bfloat16")
        do1_f, pg_f, o2s_f = FUNC(o1, do2_s, probs, clamp_value=5.0)
        do1_g, pg_g, o2s_g = _gold_clamped_bwd(o1, probs, do2_s, 5.0)
        _assert_close(
            do1_f.cast("float32").numpy(), do1_g.cast("float32").numpy()
        )
        _assert_close(pg_f.numpy(), pg_g.numpy())
        _assert_close(
            o2s_f.cast("float32").numpy(), o2s_g.cast("float32").numpy()
        )

    def test_zero_size_tensor(self):
        """0-dim tensor should not crash and produce correct output shapes."""
        mid = 128
        o1 = paddle.rand([0, 16, mid * 2], dtype="bfloat16")
        probs = paddle.rand([0, 16, 1], dtype="float32")
        do2_s = paddle.rand([0, 16, mid], dtype="bfloat16")
        do1, pg, o2_s = FUNC(o1, do2_s, probs, clamp_value=5.0)
        self.assertEqual(list(do1.shape), list(o1.shape))
        self.assertEqual(list(pg.shape), list(probs.shape))
        self.assertEqual(list(o2_s.shape), list(do2_s.shape))
        self.assertEqual(do1.numel(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
