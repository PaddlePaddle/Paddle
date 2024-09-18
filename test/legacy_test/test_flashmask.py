# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core
from paddle.nn.functional.flash_attention import (
    flashmask_attention,
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


def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    return m


def gen_random_flashmask(bz, num_head, seqlen, has_end, causal):
    mask_num = 1
    if not causal:
        mask_num *= 2
    if has_end:
        mask_num *= 2
    m = np.random.randint(0, seqlen, (bz, num_head, seqlen, mask_num))
    diag = np.arange(seqlen).reshape((1, 1, seqlen))
    m[:, :, :, 0] = np.maximum(diag + 1, m[:, :, :, 0])
    if not causal:
        if has_end:
            raise NotImplementedError
        else:
            m[:, :, :, 1] = np.minimum(diag, m[:, :, :, 1])
    else:
        if has_end:
            m[:, :, :, 1] = m[:, :, :, 0] + 1
            m[:, :, :, 1] = np.maximum(m[:, :, :, 0], m[:, :, :, 1])

    return paddle.to_tensor(m, dtype="int32")


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 7.5 or 8.x",
)
class TestFlashMaskAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = True
        self.has_end = False
        self.mask_broadcast = True

    def get_flashmask(self):
        self.startend_row_indices = gen_random_flashmask(
            self.shape[0],
            1 if self.mask_broadcast else self.shape[2],
            self.shape[1],
            self.has_end,
            self.causal,
        )
        return self.startend_row_indices

    def get_densemask(self):
        self.densemask = flashmask_to_densemask(
            self.startend_row_indices, "float32", self.causal
        )
        return self.densemask

    def get_inputs(self):
        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)
        ograd = np.random.random(self.shape)

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        ograd = paddle.to_tensor(
            ograd, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        return q, k, v, ograd

    def clone_tensors(self, *xs):
        ys = [x.detach().clone() for x in xs]
        for y in ys:
            y.stop_gradient = False
        return tuple(ys)

    def test_dot_scale_product(self):
        # test dynamic
        paddle.disable_static()

        atol = 1e-2 if self.dtype == "bfloat16" else 1e-3
        rtol = 1e-2 if self.dtype == "bfloat16" else 5e-3

        q, k, v, ograd = self.get_inputs()
        q_, k_, v_, ograd_ = self.clone_tensors(
            q.cast("float32"),
            k.cast("float32"),
            v.cast("float32"),
            ograd.cast("float32"),
        )

        startend_row_indices = self.get_flashmask()
        mask = self.get_densemask()
        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            dropout=self.dropout,
            causal=self.causal,
        )
        out_ = attention_naive_with_mask(q_, k_, v_, mask)
        out.backward(ograd)
        out_.backward(ograd_)
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            out_.cast("float32").numpy(),
            rtol=rtol,
            atol=atol,
        )
        for x, y in [(q, q_), (k, k_), (v, v_)]:
            np.testing.assert_allclose(
                x.grad.cast("float32").numpy(),
                y.grad.cast("float32").numpy(),
                rtol=rtol,
                atol=atol,
            )


class TestFlashMaskAttentionFP16API1(TestFlashMaskAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = True
        self.has_end = False
        self.mask_broadcast = True


class TestFlashMaskAttentionBF16API1(TestFlashMaskAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 128)
        self.dtype = 'bfloat16'
        self.dropout = 0.0
        self.causal = True
        self.has_end = False
        self.mask_broadcast = True


class TestFlashMaskAttentionFP16API2(TestFlashMaskAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (8, 1024, 16, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.has_end = False
        self.mask_broadcast = True


class TestFlashMaskAttentionBF16API2(TestFlashMaskAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (8, 1024, 16, 128)
        self.dtype = 'bfloat16'
        self.dropout = 0.0
        self.causal = False
        self.has_end = False
        self.mask_broadcast = True


class TestFlashMaskAttentionFP16API3(TestFlashMaskAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 2048, 16, 96)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = True
        self.has_end = False
        self.mask_broadcast = False


class TestFlashMaskAttentionBF16API3(TestFlashMaskAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 2048, 16, 96)
        self.dtype = 'bfloat16'
        self.dropout = 0.0
        self.causal = True
        self.has_end = False
        self.mask_broadcast = False
