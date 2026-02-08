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


if __name__ == '__main__':
    unittest.main()
