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

import os
import re
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.device import core
from paddle.nn.functional.flash_attention import (
    flash_attention,
    scaled_dot_product_attention,
)


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


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


is_sm80 = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] == 0
)

is_sm8x = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] >= 0
)

is_sm90 = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 9
    and paddle.device.cuda.get_device_capability()[1] == 0
)

is_sm_supported = is_sm8x or is_sm90


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestFlashAttentionAPIFlag(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 16)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False
        self.use_sdp_api = False

    def flash_attn_compute(self, query, key, value):
        # test dynamic
        paddle.disable_static()

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

        if self.use_sdp_kernel:
            with paddle.nn.functional.sdp_kernel(
                enable_math=self.enable_math,
                enable_flash=self.enable_flash,
                enable_mem_efficient=self.enable_mem_efficient,
            ):
                if self.use_sdp_api:
                    out = scaled_dot_product_attention(
                        q, k, v, None, self.dropout, self.causal
                    )
                else:
                    out, _ = flash_attention(
                        q, k, v, self.dropout, self.causal, self.return_softmax
                    )

        else:
            out, _ = flash_attention(
                q, k, v, self.dropout, self.causal, self.return_softmax
            )
        out_ = attention_naive(q_, k_, v_, self.causal)

        out.backward()
        out_.backward()

        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(q_.grad.shape, q.shape)

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-03, atol=2e-03
        )

        return out, out_, q.grad.numpy(), k.grad.numpy(), v.grad.numpy()

    def test_all_flag(self):
        paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        out1, out1_, q_grad1, k_grad1, v_grad1 = self.flash_attn_compute(
            query, key, value
        )

        np.testing.assert_allclose(out1.numpy(), out1_, rtol=5e-03, atol=1e-03)

        out2, out2_, q_grad2, k_grad2, v_grad2 = self.flash_attn_compute(
            query, key, value
        )
        self.assertTrue(np.equal(out1.numpy(), out2.numpy()).all())
        self.assertTrue(np.equal(q_grad1, q_grad2).all())
        self.assertTrue(np.equal(k_grad1, k_grad2).all())
        self.assertTrue(np.equal(v_grad1, v_grad2).all())
        paddle.set_flags({'FLAGS_cudnn_deterministic': 0})


class TestFlashAttentionAPIFlagTest1(TestFlashAttentionAPIFlag):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 16)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestFlashAttentionAPIFlagTest2(TestFlashAttentionAPIFlag):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        # Flash attention backward kernel only supports SM80 or SM90 for head dimension > 192
        self.shape = (
            (8, 1024, 16, 256) if (is_sm80 or is_sm90) else (8, 1024, 16, 192)
        )
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestSDPAttentionAPIFlagTest(TestFlashAttentionAPIFlag):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (8, 1024, 16, 128)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = True
        self.use_sdp_api = True
        self.enable_math = True
        self.enable_flash = False
        self.enable_mem_efficient = False


if __name__ == '__main__':
    unittest.main()
