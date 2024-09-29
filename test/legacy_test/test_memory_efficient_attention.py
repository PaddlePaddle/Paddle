# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import logging
import os
import random
import re
import unittest
from typing import TYPE_CHECKING

import numpy as np

import paddle
import paddle.incubate.nn.attn_bias as ab
import paddle.nn.functional as F
from paddle.base import core
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

paddle.seed(2023)


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


def create_attn_bias(
    bias_type,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    tdtype,
    pdtype,
    requires_grad: bool,
    fmt: str,
):
    if bias_type is None or isinstance(None, bias_type):
        return None
    r = random.Random(
        "-".join(map(str, [batch_size, q_len, kv_len, tdtype, fmt]))
    )
    if bias_type is paddle.Tensor:
        if fmt == "BMK":
            batch_size *= num_heads
            num_heads = 1
        attn_bias = (
            paddle.randn((batch_size, num_heads, 1, kv_len), dtype=pdtype) * 3
        )
        attn_bias = attn_bias.expand([batch_size, num_heads, q_len, kv_len])
        if requires_grad:
            attn_bias.stop_gradient = False
        return attn_bias
    if bias_type is ab.LowerTriangularMask:
        return ab.LowerTriangularMask()
    if bias_type in [
        ab.BlockDiagonalMask,
        ab.BlockDiagonalCausalMask,
    ]:
        # This bias is not supported in BMK format
        assert fmt == "BMHK"
        block_diag = ab.BlockDiagonalMask.from_seqlens(
            *_rand_seqlens(r, batch_size, q_len, kv_len)
        )
        if bias_type is ab.BlockDiagonalCausalMask:
            block_diag = block_diag.make_causal()
        return block_diag
    raise AssertionError(f"Unsupported bias type: {bias_type}")


def _rand_seqlens(
    r: random.Random, bs: int, q_len: int, kv_len: int
) -> tuple[Sequence[int], Sequence[int]]:
    q_len *= bs
    kv_len *= bs
    seqlens_q: list[int] = []
    seqlens_k: list[int] = []

    step_q = [max(1, q_len // 10), max(2, q_len // 2)]
    step_k = [max(1, kv_len // 10), max(2, kv_len // 2)]
    while sum(seqlens_q) < q_len and sum(seqlens_k) < kv_len:
        seqlens_q.append(r.randrange(*step_q))
        seqlens_k.append(r.randrange(*step_k))
    seqlens_q[-1] = q_len - sum(seqlens_q[:-1])
    seqlens_k[-1] = kv_len - sum(seqlens_k[:-1])
    return seqlens_q, seqlens_k


def attention_naive(q, k, v, attn_bias, dropout_prob, scale, seed):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)

    if attn_bias is None:
        dropout_input = F.softmax(s)
    elif isinstance(
        attn_bias,
        (
            ab.LowerTriangularMask,
            ab.BlockDiagonalMask,
            ab.BlockDiagonalCausalMask,
        ),
    ):
        bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]), q.dtype
        )
        dropout_input = F.softmax(s + bias)
    elif isinstance(attn_bias, paddle.Tensor):
        dropout_input = F.softmax(s + attn_bias)

    paddle.seed(seed)
    dropout_output = F.dropout(
        x=dropout_input,
        p=dropout_prob,
        training=True,
        mode="upscale_in_train",
    )

    o = paddle.matmul(dropout_output, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11030,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.3",
)
class TestMemEffAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.name = "MemEffAPI_fp32"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 128, 8, 16)
        self.dtype = 'float32'
        self.dropout = 0.0
        self.training = True
        self.attention_bias = None
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023

    def test_all(self):
        print(
            f"Test All case shape {self.shape} dtype {self.dtype} name {self.name}"
        )

        paddle.disable_static()

        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key = np.random.random(self.shape)
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value = np.random.random(self.shape)
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q.stop_gradient = False
        k.stop_gradient = False
        v.stop_gradient = False
        q_.stop_gradient = False
        k_.stop_gradient = False
        v_.stop_gradient = False

        out_ = attention_naive(
            q_, k_, v_, self.attention_bias, self.dropout, self.scale, self.seed
        )

        paddle.seed(self.seed)
        out = memory_efficient_attention(
            q,
            k,
            v,
            self.attention_bias,
            self.dropout,
            self.scale,
            self.training,
        )

        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

        grad_out = paddle.ones_like(q)

        out.backward(grad_out)
        out_.backward(grad_out)

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-03, atol=1e-03
        )


class TestMemEffAPIDtypeFp16(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp16"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 32, 128, 128)
        self.dtype = paddle.float16
        self.dropout = 0.0
        self.attention_bias = None
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


class TestMemEffAPIShape0(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp32"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 32, 128, 32)
        self.dtype = paddle.float32
        self.dropout = 0.0
        self.attention_bias = None
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


class TestMemEffAPIShape1(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp32"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 32, 16, 16)
        self.dtype = paddle.float32
        self.dropout = 0.0
        self.attention_bias = None
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


class TestMemEffAPIShape2(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp32"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 32, 8, 8)
        self.dtype = paddle.float32
        self.dropout = 0.0
        self.attention_bias = None
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


class TestMemEffAPIShape3(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp32"
        self.place = paddle.CUDAPlace(0)
        self.shape = (16, 32, 128, 128)
        self.dtype = paddle.float32
        self.dropout = 0.0
        self.attention_bias = None
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


class TestMemEffAPIMask0(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp32_BlockDiagonalMask"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 32, 128, 128)
        self.dtype = paddle.float32
        self.dropout = 0.0
        self.attention_bias = create_attn_bias(
            ab.BlockDiagonalMask,
            self.shape[0],
            self.shape[2],
            self.shape[1],
            self.shape[1],
            "float32",
            self.dtype,
            False,
            "BMHK",
        )
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


class TestMemEffAPIMask1(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp32_BlockDiagonalCausalMask"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 32, 128, 128)
        self.dtype = paddle.float32
        self.dropout = 0.0
        self.attention_bias = create_attn_bias(
            ab.BlockDiagonalCausalMask,
            self.shape[0],
            self.shape[2],
            self.shape[1],
            self.shape[1],
            "float32",
            self.dtype,
            False,
            "BMHK",
        )
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


class TestMemEffAPIMask2(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp32_LowerTriangularMask"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 32, 128, 128)
        self.dtype = paddle.float32
        self.dropout = 0.0
        self.attention_bias = create_attn_bias(
            ab.LowerTriangularMask,
            self.shape[0],
            self.shape[2],
            self.shape[1],
            self.shape[1],
            "float32",
            self.dtype,
            False,
            "BMHK",
        )
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


class TestMemEffAPIMask3(TestMemEffAttentionAPI):
    def setUp(self):
        self.name = "MemEffAPI_fp32_AnyTensor"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 32, 128, 128)
        self.dtype = paddle.float32
        self.dropout = 0.0
        self.attention_bias = (
            paddle.randn(
                (self.shape[0], self.shape[2], 1, self.shape[1]),
                dtype=self.dtype,
            )
            * 3
        )
        self.attention_bias = self.attention_bias.expand(
            [self.shape[0], self.shape[2], self.shape[1], self.shape[1]]
        )
        self.attention_bias.stop_gradient = False
        self.training = True
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11030,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.3",
)
class TestMemEffAttentionAPIWithStopGradient(unittest.TestCase):
    def setUp(self):
        self.name = "MemEffAttnQKV_FFF"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 128, 8, 16)
        self.dtype = 'float32'
        self.dropout = 0.0
        self.training = True
        self.attention_bias = None
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023
        self.q_grad_stop_gradient = True
        self.k_grad_stop_gradient = False
        self.v_grad_stop_gradient = False

    def test_all(self):
        logging.info(
            f"Test All case shape {self.shape} dtype {self.dtype} name {self.name}"
        )

        paddle.disable_static()

        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query,
            place=self.place,
            dtype=self.dtype,
            stop_gradient=self.q_grad_stop_gradient,
        )
        q_ = paddle.to_tensor(
            query,
            place=self.place,
            dtype=self.dtype,
            stop_gradient=self.q_grad_stop_gradient,
        )
        key = np.random.random(self.shape)
        k = paddle.to_tensor(
            key,
            place=self.place,
            dtype=self.dtype,
            stop_gradient=self.k_grad_stop_gradient,
        )
        k_ = paddle.to_tensor(
            key,
            place=self.place,
            dtype=self.dtype,
            stop_gradient=self.k_grad_stop_gradient,
        )
        value = np.random.random(self.shape)
        v = paddle.to_tensor(
            value,
            place=self.place,
            dtype=self.dtype,
            stop_gradient=self.v_grad_stop_gradient,
        )
        v_ = paddle.to_tensor(
            value,
            place=self.place,
            dtype=self.dtype,
            stop_gradient=self.v_grad_stop_gradient,
        )

        out_ = attention_naive(
            q_, k_, v_, self.attention_bias, self.dropout, self.scale, self.seed
        )

        paddle.seed(self.seed)
        out = memory_efficient_attention(
            q,
            k,
            v,
            self.attention_bias,
            self.dropout,
            self.scale,
            self.training,
        )

        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

        out.backward()
        out_.backward()

        if q.stop_gradient is not True:
            np.testing.assert_allclose(
                q.grad.numpy(), q_.grad.numpy(), rtol=5e-03, atol=1e-03
            )

        if k.stop_gradient is not True:
            np.testing.assert_allclose(
                k.grad.numpy(), k.grad.numpy(), rtol=5e-03, atol=1e-03
            )
        if v.stop_gradient is not True:
            np.testing.assert_allclose(
                v.grad.numpy(), v_.grad.numpy(), rtol=5e-03, atol=1e-03
            )


class TestQKVFTT(TestMemEffAttentionAPIWithStopGradient):
    def setUp(self):
        self.name = "MemEffAttnQKV_TTT"
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 128, 8, 16)
        self.dtype = 'float32'
        self.dropout = 0.0
        self.training = True
        self.attention_bias = None
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.seed = 2023
        self.q_grad_stop_gradient = False
        self.k_grad_stop_gradient = True
        self.v_grad_stop_gradient = True


if __name__ == '__main__':
    unittest.main()
