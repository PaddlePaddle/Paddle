# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import convert_uint16_to_float

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.nn.functional.flash_attention import (
    flash_attention,
)


def get_triangle_upper_mask(x):
    mask = paddle.full_like(x, -1e4)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def attention_naive(q_bf16, k_bf16, v_bf16):
    q = paddle.cast(q_bf16, "float32")
    k = paddle.cast(k_bf16, "float32")
    v = paddle.cast(v_bf16, "float32")

    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    mask = get_triangle_upper_mask(s)
    s = s + mask
    p = F.softmax(s)
    o = paddle.matmul(p, vt)
    o = paddle.cast(o, "bfloat16")
    return paddle.transpose(o, [0, 2, 1, 3])


def is_flashattn_supported():
    xpu_version = core.get_xpu_device_version(0)
    if xpu_version != core.XPUVersion.XPU3:
        return False
    xhpc_version = paddle.version.xpu_xhpc()
    if xhpc_version == 'False':
        return False
    return True


@unittest.skipIf(
    not is_flashattn_supported(), "only available on XPU3 with XHPC"
)
class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.shape = (1, 128, 2, 32)
        self.dtype = 'bfloat16'
        self.dropout = 0.0
        self.causal = True
        self.return_softmax = False

    def test_all(self):
        print(f"Test case shape {self.shape} dtype {self.dtype}")
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

        out, _ = flash_attention(
            q, k, v, self.dropout, self.causal, self.return_softmax
        )
        # TODO(houj04): use self.causal
        out_ = attention_naive(q_, k_, v_)

        out.backward()
        out_.backward()

        # forward result
        float_out = convert_uint16_to_float(out)
        float_out_ = convert_uint16_to_float(out_)
        np.testing.assert_allclose(
            float_out, float_out_, rtol=1e-02, atol=1e-02
        )

        # backward shape
        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(q_.grad.shape, q.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(k_.grad.shape, k.shape)
        self.assertEqual(v.grad.shape, v.shape)
        self.assertEqual(v_.grad.shape, v.shape)

        # backward result
        float_q_grad = convert_uint16_to_float(q.grad)
        float_q_grad_ = convert_uint16_to_float(q_.grad)
        float_k_grad = convert_uint16_to_float(k.grad)
        float_k_grad_ = convert_uint16_to_float(k_.grad)
        float_v_grad = convert_uint16_to_float(v.grad)
        float_v_grad_ = convert_uint16_to_float(v_.grad)

        # TODO(houj04) fix ut threshold
        np.testing.assert_allclose(
            float_q_grad, float_q_grad_, rtol=1e-01, atol=1e-01
        )
        np.testing.assert_allclose(
            float_k_grad, float_k_grad_, rtol=1e-01, atol=1e-01
        )
        np.testing.assert_allclose(
            float_v_grad, float_v_grad_, rtol=1e-01, atol=1e-01
        )


if __name__ == '__main__':
    unittest.main()
