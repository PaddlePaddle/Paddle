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


def allclose(x, y):
    np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=1e-5)


_TEST_PROBLEMS = (
    (1, 128, 128, 128),
    (8, 128, 128, 128),
    (16, 128, 128, 128),
    (1, 128, 256, 512),
    (8, 128, 256, 512),
    (16, 128, 256, 512),
)

_TRANS_CASES = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)


def randn(bs, x, y):
    out = (paddle.rand([bs, x, y]) - 0.5 * 2) / (y * x)
    return out.astype(paddle.bfloat16)


def pyref_gmm(a, b, batch_sizes, trans_a=False, trans_b=False):
    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        lhs = a[i, :, :].t() if trans_a else a[start : start + size, :]
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(lhs @ rhs)
        if not trans_a:
            start += size
    if trans_a:
        return paddle.stack(out, axis=0)
    return paddle.concat(out, axis=0)


class TestGroupedGemm(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)

    def _run_test(self, z, m, k, n, batch_sizes, trans_a, trans_b):
        with paddle.amp.auto_cast(False):
            # Prepare inputs based on transpose flags
            if trans_a:
                # Case 3: [M_total, K]' x [M_total, N] -> [z, K, N]
                a = randn(z, m, k).reshape([-1, k]).astype(paddle.bfloat16)
                b = randn(1, m * z, n).reshape([-1, n]).astype(paddle.bfloat16)
                a_ref = randn(z, k, m).astype(
                    paddle.bfloat16
                )  # For pyref: [z, K, M]
                b_ref = randn(z, m, n).astype(
                    paddle.bfloat16
                )  # Rebuild for pyref
                # Rebuild a_ref and b_ref from actual data
                start = 0
                a_ref_list, b_ref_list = [], []
                for i, size in enumerate(batch_sizes):
                    a_ref_list.append(
                        a[start : start + size, :].t().unsqueeze(0)
                    )
                    b_ref_list.append(b[start : start + size, :].unsqueeze(0))
                    start += size
                a_ref = paddle.concat(a_ref_list, axis=0)
                b_ref = paddle.concat(b_ref_list, axis=0)
            else:
                # Case 1 & 2: [M_total, K] x [z, K, N] -> [M_total, N]
                a = randn(z, m, k).reshape([-1, k]).astype(paddle.bfloat16)
                if trans_b:
                    b = randn(z, n, k).astype(
                        paddle.bfloat16
                    )  # Will be transposed
                else:
                    b = randn(z, k, n).astype(paddle.bfloat16)
                a_ref = a.clone().detach()
                b_ref = b.clone().detach()

            a.stop_gradient = False
            b.stop_gradient = False

            out = grouped_gemm(a, b, batch_sizes, trans_a, trans_b)
            expected_out = pyref_gmm(
                a_ref, b_ref, batch_sizes, trans_a, trans_b
            )
            allclose(out, expected_out)

    def test_grouped_gemm_fixed_sizes(self):
        """Test grouped GEMM with fixed sizes and all transpose combinations"""
        for z, m, k, n in _TEST_PROBLEMS:
            for trans_a, trans_b in _TRANS_CASES:
                with self.subTest(
                    z=z, m=m, k=k, n=n, trans_a=trans_a, trans_b=trans_b
                ):
                    batch_sizes = [m] * z
                    self._run_test(z, m, k, n, batch_sizes, trans_a, trans_b)

    def test_grouped_gemm_variable_sizes(self):
        """Test grouped GEMM with variable sizes and all transpose combinations"""
        for z, m, k, n in _TEST_PROBLEMS:
            for trans_a, trans_b in _TRANS_CASES:
                with self.subTest(
                    z=z, m=m, k=k, n=n, trans_a=trans_a, trans_b=trans_b
                ):
                    dist = paddle.rand([z])
                    dist /= dist.sum()
                    batch_sizes = (dist * m).astype(paddle.int64)
                    batch_sizes[-1] += m * z - batch_sizes.sum()
                    batch_sizes = [int(x) for x in batch_sizes]
                    self._run_test(z, m, k, n, batch_sizes, trans_a, trans_b)


if __name__ == '__main__':
    unittest.main()
