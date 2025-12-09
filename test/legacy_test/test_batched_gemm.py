# !/usr/bin/env python3

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
    mask = np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=1e-5)


_TEST_PROBLEMS = (
    (1, 128, 128, 128),
    (8, 128, 128, 128),
    (16, 128, 128, 128),
    (1, 128, 256, 512),
    (8, 128, 256, 512),
    (16, 128, 256, 512),
)


def randn(bs, x, y):
    out = (paddle.rand([bs, x, y]) - 0.5 * 2) / (y * x)
    return out.astype(paddle.bfloat16)


def pyref_gmm(a, b, batch_sizes, trans_b=False):
    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return paddle.concat(out, axis=0)


class TestGroupedGemm(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)

    def test_grouped_gemm_fixed_sizes(self):
        """Test grouped GEMM with fixed sizes"""
        for z, m, k, n in _TEST_PROBLEMS:
            with self.subTest(
                z=z, m=m, k=k, n=n, trans_b=False
            ) and paddle.amp.auto_cast(False):
                a = randn(z, m, k).reshape([-1, k]).astype(paddle.bfloat16)
                b = randn(z, k, n).astype(paddle.bfloat16)
                batch_sizes = [m] * z
                a.stop_gradient = False
                b.stop_gradient = False
                a_ref = a.clone().detach()
                b_ref = b.clone().detach()
                a_ref.stop_gradient = False
                b_ref.stop_gradient = False

                out = grouped_gemm(a, b, batch_sizes)
                expected_out = pyref_gmm(a_ref, b_ref, batch_sizes, False)
                allclose(out, expected_out)

    def test_grouped_gemm_variable_sizes(self):
        """Test grouped GEMM with variable sizes"""
        for z, m, k, n in _TEST_PROBLEMS:
            with self.subTest(
                z=z, m=m, k=k, n=n, trans_b=False
            ) and paddle.amp.auto_cast(False):
                trans_b = False
                a = randn(z, m, k).reshape([-1, k]).astype(paddle.bfloat16)
                b = randn(z, k, n).astype(paddle.bfloat16)

                dist = paddle.rand([z])
                dist /= dist.sum()
                batch_sizes = (dist * m).astype(paddle.int64)
                error = m * z - batch_sizes.sum()
                batch_sizes[-1] += error
                if batch_sizes.sum() != m * z:
                    raise ValueError("Sum of batch sizes is not equal to m * z")
                batch_sizes = list(batch_sizes)

                a.stop_gradient = False
                b.stop_gradient = False
                a_ref = a.clone().detach()
                b_ref = b.clone().detach()
                a_ref.stop_gradient = False
                b_ref.stop_gradient = False

                out = grouped_gemm(a, b, batch_sizes)
                expected_out = pyref_gmm(a_ref, b_ref, batch_sizes, trans_b)
                allclose(out, expected_out)


if __name__ == '__main__':
    unittest.main()
