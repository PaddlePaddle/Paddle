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
from paddle import base
from paddle.base import core
from paddle.nn.functional.flash_attention import (
    flash_attention,
    flash_attn_unpadded,
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
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
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
class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 16)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False
        self.use_sdp_api = False

    def test_unpadded(self):
        print(
            f"Test unpadded case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        )

        paddle.disable_static()

        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        out_ = attention_naive(q_, q_, q_, self.causal)

        scale = 1.0 / np.sqrt(q.shape[-1])

        bs = self.shape[0]
        ms = self.shape[1]
        nh = self.shape[2]
        hd = self.shape[3]
        cu_q = paddle.arange(0, (bs + 1) * ms, ms, dtype='int32')

        qq = paddle.reshape(q, [bs * ms, nh, hd])
        out, _ = flash_attn_unpadded(
            qq,
            qq,
            qq,
            cu_q,
            cu_q,
            ms,
            ms,
            scale,
            self.dropout,
            self.causal,
            self.return_softmax,
        )
        out_ = paddle.reshape(out_, [bs * ms, nh, hd])

        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

        out.backward()
        out_.backward()

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-03, atol=1e-03
        )

        # test static
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            qs = paddle.static.data(
                name="q", shape=self.shape, dtype=self.dtype
            )

            cu_q = paddle.arange(0, (bs + 1) * ms, ms, dtype='int32')
            qs = paddle.reshape(qs, [bs * ms, nh, hd])

            outs, softmax = flash_attn_unpadded(
                qs,
                qs,
                qs,
                cu_q,
                cu_q,
                ms,
                ms,
                scale,
                self.dropout,
                self.causal,
                self.return_softmax,
            )

            exe = base.Executor(self.place)
            fetches_result = exe.run(
                feed={
                    "q": query.astype('float16'),
                    "k": query.astype('float16'),
                    "v": query.astype('float16'),
                },
                fetch_list=[outs],
            )

            np.testing.assert_allclose(
                fetches_result[0], out_, rtol=5e-03, atol=1e-03
            )

    def test_all(self):
        print(
            f"Test case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        )
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

        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(q_.grad.shape, q.shape)

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-03, atol=1e-03
        )

        # test static
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            qs = paddle.static.data(
                name="q", shape=self.shape, dtype=self.dtype
            )
            ks = paddle.static.data(
                name="k", shape=self.shape, dtype=self.dtype
            )
            vs = paddle.static.data(
                name="v", shape=self.shape, dtype=self.dtype
            )

            if self.use_sdp_kernel:
                with paddle.nn.functional.sdp_kernel(
                    enable_math=self.enable_math,
                    enable_flash=self.enable_flash,
                    enable_mem_efficient=self.enable_mem_efficient,
                ):
                    if self.use_sdp_api:
                        outs = scaled_dot_product_attention(
                            qs, ks, vs, None, self.dropout, self.causal
                        )
                    else:
                        outs, softmax = flash_attention(
                            qs,
                            ks,
                            vs,
                            self.dropout,
                            self.causal,
                            self.return_softmax,
                        )
            else:
                outs, softmax = flash_attention(
                    qs, ks, vs, self.dropout, self.causal, self.return_softmax
                )

            exe = base.Executor(self.place)
            fetches_result = exe.run(
                feed={
                    "q": query.astype('float16'),
                    "k": key.astype('float16'),
                    "v": value.astype('float16'),
                },
                fetch_list=[outs],
            )

            np.testing.assert_allclose(
                fetches_result[0], out_, rtol=5e-03, atol=1e-03
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 7.5 or 8.x",
)
class TestFlashAttentionWithMaskAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 32)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False

    def test_dot_scale_product(self):
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
        mask = np.random.random(mask_shape)
        m = paddle.to_tensor(
            mask, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        out = scaled_dot_product_attention(
            q, k, v, m, self.dropout, self.causal
        )
        out_ = attention_naive_with_mask(q_, k_, v_, m)
        out.backward()
        out_.backward()
        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)


class TestFlashAttentionAPITest1(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 16)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestFlashAttentionAPITest2(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 256, 8, 16)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = True
        self.use_sdp_kernel = False


class TestFlashAttentionAPITest3(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 512, 8, 16)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = True
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestFlashAttentionAPITest4(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (8, 1024, 16, 128)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestFlashAttentionAPITest5(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (8, 1024, 16, 256)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestMathAttentionAPITest(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (8, 1024, 16, 128)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = True
        self.use_sdp_api = False
        self.enable_math = True
        self.enable_flash = False
        self.enable_mem_efficient = False


class TestSDPAttentionAPITest(TestFlashAttentionAPI):
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


class TestFlashAttrnionWithMaskAPI(TestFlashAttentionWithMaskAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (8, 1024, 16, 128)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False


if __name__ == '__main__':
    unittest.main()
