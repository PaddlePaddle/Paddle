#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np

import paddle
from paddle.incubate.nn.functional import fused_partial_rope


def fused_partial_rope_ref(x, cos, sin):
    x_nope = x[..., : -cos.shape[-1]]
    x_pe = x[..., -cos.shape[-1] :]

    b, s, h, d = x_pe.shape  # [bs, seq_len, num_heads, pe_head_dim]
    x_pe = (
        x_pe.reshape([b, s, h, d // 2, 2])
        .transpose([0, 1, 2, 4, 3])
        .reshape([b, s, h, d])
    )

    cos = cos[:, :s, :, :]  # [1, seq_len, 1, pe_head_dim]
    sin = sin[:, :s, :, :]

    x1 = x_pe[..., : x_pe.shape[-1] // 2]
    x2 = x_pe[..., x_pe.shape[-1] // 2 :]
    x_pe_rotate_half = paddle.concat([-x2, x1], axis=-1)

    x_pe = (x_pe * cos) + (x_pe_rotate_half * sin)

    return paddle.concat([x_nope, x_pe], axis=-1)


class TestFusedPartialRoPEOp(unittest.TestCase):
    def eval(self, batch_size, seq_len, num_heads, head_dim, pe_head_dim):
        x = paddle.randn([batch_size, seq_len, num_heads, head_dim], 'bfloat16')
        x.stop_gradient = False
        x_ref = paddle.clone(x).detach()
        x_ref.stop_gradient = False

        cos = paddle.randn([1, seq_len, 1, pe_head_dim], 'bfloat16')
        sin = paddle.randn_like(cos)

        # Test forward
        out = fused_partial_rope(x, cos, sin)
        out_ref = fused_partial_rope_ref(x_ref, cos, sin)

        np.testing.assert_allclose(
            out.astype('float32'), out_ref.astype('float32')
        )

        # Test backward
        out_grad = paddle.randn_like(out)
        paddle.autograd.backward([out], [out_grad])
        paddle.autograd.backward([out_ref], [out_grad])

        np.testing.assert_allclose(
            x.grad.astype('float32'), x_ref.grad.astype('float32')
        )

    def test_0_size_in_batch_size(self):
        self.eval(0, 32, 64, 128, 64)

    def test_0_size_in_seq_len(self):
        self.eval(32, 0, 64, 128, 64)

    def test_all_pe_head_dim(self):
        self.eval(1, 8, 1, 128, 128)

    def test_medium_1x_vec(self):
        self.eval(1, 8, 16, 75, 50)

    def test_medium_2x_vec(self):
        self.eval(4, 1, 16, 200, 100)

    def test_medium_4x_vec(self):
        self.eval(2, 4, 8, 192, 64)

    def test_large(self):
        self.eval(1, 2, 16, 1024, 384)


if __name__ == "__main__":
    unittest.main()
