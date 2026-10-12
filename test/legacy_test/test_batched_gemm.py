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

import os
import unittest

import numpy as np

import paddle
from paddle.incubate.nn.functional import batched_gemm as grouped_gemm

os.environ["FLAGS_flash_attn_version"] = "v1"
os.environ["FLAGS_cudnn_deterministic"] = "1"
os.environ["FLAGS_embedding_deterministic"] = "1"


def allclose(x, y, dtype):
    if dtype == paddle.bfloat16:
        rtol = 1e-5
    else:
        rtol = 1e-5
    np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=rtol)


_TEST_PROBLEMS = (
    (1, 128, 128, 128),
    (8, 128, 128, 128),
    (16, 128, 128, 128),
    (1, 128, 256, 512),
    (8, 128, 256, 512),
    (16, 128, 256, 512),
)

m_group_layout_cases = [(False, True), (False, False)]


def randn(bs, x, y, dtype=paddle.bfloat16):
    out = (paddle.rand([bs, x, y]) - 0.5 * 2) / (y * x)
    return out.astype(dtype)


def pyref_gmm(a, b, batch_sizes, trans_b=False):
    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        lhs = a[start : start + size, :]
        rhs = b[i, :, :] if not trans_b else b[i, :, :].t()
        out.append(lhs @ rhs)
        start += size
    return paddle.concat(out, axis=0)


def pyref_k_gmm(a, b, batch_sizes):
    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        lhs = a[start : start + size, :].t()
        rhs = b[start : start + size, :]
        out.append(lhs @ rhs)
        start += size
    return paddle.concat(out, axis=0)


class TestGroupedGemm(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)

    def test_m_grouped_gemm_fixed_sizes(self):
        """Test grouped GEMM with fixed sizes"""
        # Test both bfloat16 and float32 dtypes
        dtypes = [paddle.bfloat16, paddle.float32]

        for dtype in dtypes:
            for z, m, k, n in _TEST_PROBLEMS:
                for trans_lhs, trans_rhs in m_group_layout_cases:
                    with self.subTest(
                        dtype=dtype,
                        z=z,
                        m=m,
                        k=k,
                        n=n,
                        trans_a=trans_lhs,
                        trans_b=trans_rhs,
                    ) and paddle.amp.auto_cast(False):
                        a = randn(z, m, k, dtype).reshape([-1, k]).astype(dtype)
                        b = randn(z, k, n, dtype).astype(dtype)
                        if trans_rhs:
                            b = b.mT
                        batch_sizes = [m] * z
                        a.stop_gradient = False
                        b.stop_gradient = False
                        a_ref = a.clone().detach()
                        b_ref = b.clone().detach()
                        a_ref.stop_gradient = False
                        b_ref.stop_gradient = False
                        print(
                            f"Testing dtype={dtype}, shape={a.shape}, {b.shape}"
                        )
                        out = grouped_gemm(a, b, batch_sizes, False, trans_rhs)
                        expected_out = pyref_gmm(
                            a_ref, b_ref, batch_sizes, trans_rhs
                        )
                        allclose(out, expected_out.reshape(out.shape), dtype)

    def test_k_grouped_gemm_variable_sizes(self):
        """Test grouped GEMM with variable sizes"""
        # Test both bfloat16 and float32 dtypes
        dtypes = [paddle.bfloat16, paddle.float32]

        for dtype in dtypes:
            for z, m, k, n in _TEST_PROBLEMS:
                with self.subTest(
                    dtype=dtype, z=z, m=m, k=k, n=n, trans_a=True, trans_b=False
                ) and paddle.amp.auto_cast(False):
                    a = randn(z, m, k, dtype).astype(dtype)
                    b = randn(z, m, n, dtype).astype(dtype)

                    batch_sizes = [m] * z

                    a.stop_gradient = False
                    b.stop_gradient = False
                    a_ref = a.clone().detach()
                    b_ref = b.clone().detach()
                    a_ref.stop_gradient = False
                    b_ref.stop_gradient = False

                    out = grouped_gemm(
                        a.reshape([-1, k]),
                        b.reshape([-1, n]),
                        batch_sizes,
                        True,
                        False,
                    )
                    expected_out = pyref_k_gmm(
                        a_ref.reshape([-1, k]),
                        b_ref.reshape([-1, n]),
                        batch_sizes,
                    )
                    allclose(out, expected_out.reshape(out.shape), dtype)


class TestGroupedGemmEdgeCases(unittest.TestCase):
    """Regression tests for boundary / degenerate shapes of batched_gemm.

    Covers the InferMeta shape validations (expert count, contraction dim,
    batch_sizes sum / negativity), the legal empty-contraction (K=0) and
    zero-token fill-zero paths, and the single-column (N == 1) grouped GEMV
    path that is aligned with torch.matmul.
    """

    def setUp(self):
        paddle.seed(42)

    def _gemm(self, a, b, batch_sizes, trans_lhs, trans_rhs):
        with paddle.amp.auto_cast(False):
            return grouped_gemm(a, b, batch_sizes, trans_lhs, trans_rhs)

    # ---- invalid inputs are rejected with InvalidArgument --------------

    def test_invalid_expert_count(self):
        # rhs[0] = 3 but batch_sizes has 4 groups; all other dims consistent.
        a = randn(1, 8, 4).reshape([-1, 4]).astype(paddle.float32)
        b = randn(3, 4, 5).astype(paddle.float32)
        with self.assertRaisesRegex(Exception, "number of groups"):
            self._gemm(a, b, [2, 2, 2, 2], False, False)

    def test_invalid_contraction_dim(self):
        # Case 1 (F, F): lhs[1] = 4 != rhs[1] = 6.
        a = randn(1, 8, 4).reshape([-1, 4]).astype(paddle.float32)
        b = randn(4, 6, 5).astype(paddle.float32)
        with self.assertRaisesRegex(Exception, "contraction dim"):
            self._gemm(a, b, [2, 2, 2, 2], False, False)

    def test_invalid_contraction_dim_trans_rhs(self):
        # Case 2 (F, T): lhs[1] = 5 != rhs[2] = 7.
        a = randn(1, 8, 5).reshape([-1, 5]).astype(paddle.float32)
        b = randn(4, 6, 7).astype(paddle.float32)
        with self.assertRaisesRegex(Exception, "contraction dim"):
            self._gemm(a, b, [2, 2, 2, 2], False, True)

    def test_invalid_batch_sizes_sum(self):
        # sum(batch_sizes) = 6 != total_tokens = 7.
        a = randn(1, 7, 4).reshape([-1, 4]).astype(paddle.float32)
        b = randn(3, 4, 5).astype(paddle.float32)
        with self.assertRaisesRegex(Exception, "total number of tokens"):
            self._gemm(a, b, [2, 2, 2], False, False)

    def test_negative_batch_sizes(self):
        a = randn(1, 6, 4).reshape([-1, 4]).astype(paddle.float32)
        b = randn(3, 4, 5).astype(paddle.float32)
        with self.assertRaisesRegex(Exception, "non-negative"):
            self._gemm(a, b, [-1, 3, 4], False, False)

    # ---- legal empty contraction / zero tokens return all zeros --------

    def test_empty_contraction_returns_zeros(self):
        for dtype in [paddle.bfloat16, paddle.float32]:
            for trans_lhs, trans_rhs in m_group_layout_cases:
                with self.subTest(
                    dtype=dtype, trans_lhs=trans_lhs, trans_rhs=trans_rhs
                ):
                    batch_sizes = [2, 3, 3]
                    if not trans_rhs:
                        # K = 0 on both sides: lhs=[8, 0], rhs=[3, 0, 5].
                        a = randn(1, 8, 0).reshape([-1, 0]).astype(dtype)
                        b = randn(3, 0, 5).astype(dtype)
                    else:
                        # Case 2: lhs[1] = rhs[2] = 0, rhs=[3, 5, 0].
                        a = randn(1, 8, 0).reshape([-1, 0]).astype(dtype)
                        b = randn(3, 5, 0).astype(dtype)
                    out = self._gemm(a, b, batch_sizes, trans_lhs, trans_rhs)
                    np.testing.assert_array_equal(
                        out.astype(paddle.float32).numpy(),
                        np.zeros(out.shape, dtype=np.float32),
                    )

    def test_zero_tokens_returns_zeros(self):
        # Case 3 (T, F) with M_total = sum(batch_sizes) = 0: the contraction
        # dim is total_tokens, so the output [E, K, N] must be all zeros.
        for dtype in [paddle.bfloat16, paddle.float32]:
            with self.subTest(dtype=dtype):
                a = randn(1, 0, 4).reshape([-1, 4]).astype(dtype)
                b = randn(1, 0, 5).reshape([-1, 5]).astype(dtype)
                out = self._gemm(a, b, [0, 0, 0], True, False)
                self.assertEqual(out.shape, [3, 4, 5])
                np.testing.assert_array_equal(
                    out.astype(paddle.float32).numpy(),
                    np.zeros([3, 4, 5], dtype=np.float32),
                )

    # ---- single-column (N == 1) grouped GEMV ----------------------------

    def test_single_column_grouped_gemv(self):
        # When output_hidden == 1 each group is a matrix-vector product. The
        # kernel routes it through the PyTorch-aligned GEMV orientation, so
        # the result must match the per-group reference bitwise for both
        # bfloat16 and float32 (verified against torch.matmul and
        # paddle.matmul): a regression to the misaligned offset-slice writes
        # or the swapped orientation shows up as a 1-ulp divergence.
        problems = (
            (3, [2, 3, 3], 16),
            (4, [1, 5, 6, 2], 64),
            (3, [2, 8, 6], 5),
        )
        for dtype in [paddle.bfloat16, paddle.float32]:
            for num_experts, batch_sizes, hidden in problems:
                for trans_lhs, trans_rhs in m_group_layout_cases:
                    with self.subTest(
                        dtype=dtype,
                        num_experts=num_experts,
                        batch_sizes=batch_sizes,
                        hidden=hidden,
                        trans_lhs=trans_lhs,
                        trans_rhs=trans_rhs,
                    ):
                        total = sum(batch_sizes)
                        if not trans_rhs:
                            # Case 1: rhs = [E, K, 1], out = [total, 1].
                            a = (
                                randn(1, total, hidden)
                                .reshape([-1, hidden])
                                .astype(dtype)
                            )
                            b = randn(num_experts, hidden, 1).astype(dtype)
                        else:
                            # Case 2: rhs = [E, 1, N], out = [total, 1].
                            a = (
                                randn(1, total, hidden)
                                .reshape([-1, hidden])
                                .astype(dtype)
                            )
                            b = randn(num_experts, 1, hidden).astype(dtype)
                        out = self._gemm(
                            a, b, batch_sizes, trans_lhs, trans_rhs
                        )
                        self.assertEqual(list(out.shape), [total, 1])
                        expected = pyref_gmm(
                            a, b, batch_sizes, trans_rhs
                        ).reshape(out.shape)
                        np.testing.assert_array_equal(
                            out.astype(paddle.float32).numpy(),
                            expected.astype(paddle.float32).numpy(),
                        )


if __name__ == '__main__':
    unittest.main()
