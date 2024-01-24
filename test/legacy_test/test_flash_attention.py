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

import logging
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
from paddle.pir_utils import test_with_pir_api

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


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


def is_flashattn_supported():
    if (
        not core.is_compiled_with_cuda()
        or get_cuda_version() < 11040
        or not is_sm_supported
    ):
        return False
    return True


@unittest.skipIf(
    not is_flashattn_supported(),
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

        paddle.disable_static()

    @test_with_pir_api
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

        paddle.disable_static()


@unittest.skipIf(
    not is_flashattn_supported(),
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
        self.return_softmax = False
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


class TestFlashAttenionWithMaskAPITest(TestFlashAttentionWithMaskAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (8, 1024, 16, 128)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.causal = False


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 7.5 or 8.x",
)
class TestFlashAttentionNoKVGrad(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 16)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = True
        self.return_softmax = False
        self.enable_math = False
        self.enable_flash = True
        self.enable_mem_efficient = False

    def _init_tensor_from_numpy(self, array, stop_gradient):
        t = paddle.to_tensor(
            array,
            place=self.place,
            dtype=self.dtype,
            stop_gradient=stop_gradient,
        )
        return t

    def test_all(self):
        logging.info(
            f"Test case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        )
        # test dynamic
        paddle.disable_static()

        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        q = self._init_tensor_from_numpy(query, stop_gradient=False)
        k = self._init_tensor_from_numpy(key, stop_gradient=True)
        v = self._init_tensor_from_numpy(value, stop_gradient=True)

        q_ = self._init_tensor_from_numpy(query, stop_gradient=False)
        k_ = self._init_tensor_from_numpy(key, stop_gradient=True)
        v_ = self._init_tensor_from_numpy(value, stop_gradient=True)

        with paddle.nn.functional.sdp_kernel(
            enable_math=self.enable_math,
            enable_flash=self.enable_flash,
            enable_mem_efficient=self.enable_mem_efficient,
        ):
            out = scaled_dot_product_attention(
                q, k, v, None, self.dropout, self.causal
            )

        out_ = attention_naive(q_, k_, v_, self.causal)
        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

        out.backward()
        out_.backward()

        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(q_.grad.shape, q.shape)
        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-03, atol=1e-03
        )


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 7.5 or 8.x",
)
class TestFlashAttentionGQA(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 8192
        self.head_dim = 128
        self.num_group = 2
        self.dtype = paddle.bfloat16

    def gen_unpadded_data(self, dtype):
        seq_len_q = np.random.randint(
            low=1, high=self.seq_len, size=[self.batch_size]
        )
        seq_len_k = np.random.randint(
            low=1, high=self.seq_len, size=[self.batch_size]
        )
        cu_seqlen_q = paddle.to_tensor(
            [0] + np.cumsum(seq_len_q).tolist(), dtype=paddle.int32
        )
        cu_seqlen_k = paddle.to_tensor(
            [0] + np.cumsum(seq_len_k).tolist(), dtype=paddle.int32
        )

        qs, ks, vs = [], [], []
        for i in range(self.batch_size):
            tmp_q = (
                paddle.randn(
                    [seq_len_q[i] * self.num_head * self.head_dim], dtype=dtype
                )
                / 1e2
            )
            tmp_k = (
                paddle.randn(
                    [
                        seq_len_k[i]
                        * self.num_head
                        * self.head_dim
                        // self.num_group
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            tmp_v = (
                paddle.randn(
                    [
                        seq_len_k[i]
                        * self.num_head
                        * self.head_dim
                        // self.num_group
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            qs.append(tmp_q)
            ks.append(tmp_k)
            vs.append(tmp_v)

        q = paddle.concat(qs, axis=0).reshape(
            [-1, self.num_head, self.head_dim]
        )
        k = paddle.concat(ks, axis=0).reshape(
            [-1, self.num_head // self.num_group, self.head_dim]
        )
        v = paddle.concat(vs, axis=0).reshape(
            [-1, self.num_head // self.num_group, self.head_dim]
        )
        return q, k, v, cu_seqlen_q, cu_seqlen_k

    def gen_test_data(self, dtype, use_unpadded):
        assert self.num_head % self.num_group == 0
        if use_unpadded:
            q, k, v, cu_seqlen_q, cu_seqlen_k = self.gen_unpadded_data(dtype)
        else:
            q = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            k = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head // self.num_group,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            v = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head // self.num_group,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            cu_seqlen_q = None
            cu_seqlen_k = None
        out_grad = paddle.randn(q.shape, dtype=dtype) / 1e2
        return q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad

    def clone_tensor(self, tensor):
        if tensor is None:
            return None
        elif isinstance(tensor, (list, tuple)):
            return [self.clone_tensor(t) for t in tensor]
        else:
            tensor = tensor.detach().clone()
            tensor.stop_gradient = False
            return tensor

    @paddle.no_grad()
    def convert_dtype(self, tensors):
        ret = []
        for t in tensors:
            if t.dtype in [paddle.float16, paddle.bfloat16]:
                t = t.astype(paddle.float32)
            t = t.numpy()
            ret.append(t)
        return ret

    def calc_fa(
        self, q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad, causal, use_unpadded
    ):
        q, k, v = self.clone_tensor([q, k, v])
        if use_unpadded:
            scale = self.head_dim ** (-0.5)
            out = flash_attn_unpadded(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlen_q,
                cu_seqlens_k=cu_seqlen_k,
                max_seqlen_q=self.seq_len,
                max_seqlen_k=self.seq_len,
                scale=scale,
                causal=causal,
            )
        else:
            out = flash_attention(q, k, v, causal=causal)
        out = out[0]
        out.backward(out_grad)
        return self.convert_dtype([out, q.grad, k.grad, v.grad])

    def calc_raw_attn(
        self, q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad, causal, use_unpadded
    ):
        q, k, v = self.clone_tensor([q, k, v])
        if use_unpadded:
            qq, q_mask = self.pad(q, cu_seqlen_q, self.seq_len)
            kk, k_mask = self.pad(k, cu_seqlen_k, self.seq_len)
            vv, _ = self.pad(v, cu_seqlen_k, self.seq_len)
            qk_mask = paddle.matmul(q_mask, k_mask, transpose_y=True)
            qk_mask = qk_mask.reshape(
                [self.batch_size, 1, self.seq_len, self.seq_len]
            )
            qk_mask[qk_mask == 0] = -1e6
            qk_mask[qk_mask == 1] = 0
        else:
            qq, kk, vv = q, k, v

        assert len(qq.shape) == 4, qq.shape
        assert len(kk.shape) == 4, kk.shape
        assert len(vv.shape) == 4, vv.shape
        perm = [0, 2, 1, 3]
        qq = paddle.transpose(qq, perm)
        kk = paddle.transpose(kk, perm)
        kk = paddle.stack([kk] * self.num_group, axis=2).reshape(qq.shape)
        vv = paddle.transpose(vv, perm)
        vv = paddle.stack([vv] * self.num_group, axis=2).reshape(qq.shape)
        scale = self.head_dim ** (-0.5)
        weight = paddle.matmul(qq * scale, kk, transpose_y=True)
        if use_unpadded:
            weight += qk_mask
        if causal:
            shape = weight.shape[-2:]
            mask = paddle.full(shape, -np.inf, dtype=weight.dtype)
            mask = paddle.triu(mask, diagonal=1)
            weight += mask

        weight = weight.astype(paddle.float32)
        weight = F.softmax(weight)
        out = paddle.matmul(weight.astype(vv.dtype), vv)
        out = paddle.transpose(out, perm)
        if use_unpadded:
            out = self.unpad(out, cu_seqlen_q)
        out.backward(out_grad)
        return self.convert_dtype([out, q.grad, k.grad, v.grad])

    def pad(self, x, cu_seqlen, max_seqlen):
        cu_seqlen_cpu = cu_seqlen.numpy()
        split_sections = []
        for i in range(len(cu_seqlen_cpu) - 1):
            split_sections.append(cu_seqlen_cpu[i + 1] - cu_seqlen_cpu[i])

        tmp_xs = paddle.split(x, split_sections)
        batch_size = len(tmp_xs)
        tmp_masks = []
        tmp_x_pads = []
        for i in range(batch_size):
            tmp_mask = paddle.ones([max_seqlen], dtype=x.dtype)
            tmp_mask[split_sections[i] :] = 0
            tmp_mask = tmp_mask.reshape([1, -1, 1])
            tmp_masks.append(tmp_mask)

            tmp_shape = tmp_xs[i].shape
            tmp_pad = paddle.zeros(
                [max_seqlen - tmp_shape[0]] + list(tmp_shape[1:]), dtype=x.dtype
            )
            tmp_x = paddle.concat([tmp_xs[i], tmp_pad]).unsqueeze(0)
            tmp_x_pads.append(tmp_x)

        x_pad = paddle.concat(tmp_x_pads)
        mask = paddle.concat(tmp_masks)
        return x_pad, mask

    def unpad(self, x, cu_seqlen):
        cu_seqlen_cpu = cu_seqlen.numpy()
        xs = paddle.split(x, x.shape[0])
        tmp_xs = []
        for i in range(len(cu_seqlen_cpu) - 1):
            tmp = xs[i].squeeze(0)[: cu_seqlen_cpu[i + 1] - cu_seqlen_cpu[i]]
            tmp_xs.append(tmp)
        unpad_x = paddle.concat(tmp_xs)
        return unpad_x

    def test_main(self):
        for causal in [False, True]:
            for use_unpadded in [False, True]:
                (
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                ) = self.gen_test_data(self.dtype, use_unpadded)
                fa_out = self.calc_fa(
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                    causal,
                    use_unpadded,
                )
                raw_out = self.calc_raw_attn(
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                    causal,
                    use_unpadded,
                )
                assert len(fa_out) == len(raw_out)
                for t1, t2 in zip(fa_out, raw_out):
                    np.testing.assert_allclose(t1, t2, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
