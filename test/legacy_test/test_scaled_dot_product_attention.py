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

import paddle
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import (
    scaled_dot_product_attention,
    sdp_kernel,
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
    not paddle.is_compiled_with_cuda(),
    "CUDA is not available, this test requires GPU support.",
)
class TestAttentionWithBoolMask(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 1, 8, 8)
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


if __name__ == '__main__':
    unittest.main()
