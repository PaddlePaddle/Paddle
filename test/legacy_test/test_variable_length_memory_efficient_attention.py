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
from paddle import base
from paddle.framework import core
from paddle.incubate.nn.functional import (
    variable_length_memory_efficient_attention,
)

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


def create_attn_mask(
    mask_type,
    batch_size,
    seq_lens,
):
    max_seq_len = max(seq_lens)
    mask = paddle.zeros(
        [batch_size, 1, max_seq_len, max_seq_len], dtype=mask_type
    )
    for i in range(batch_size):
        seq_len = seq_lens[i]
        mask[i, 0, :seq_len, :seq_len] = (
            paddle.tril(paddle.ones(shape=(seq_len, seq_len), dtype=mask_type))
            - 1
        ) * 1e4
    return mask


def naive_attention_impl(query, key, value, mask, scale):
    batch = query.shape[0]
    heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    kv_head = key.shape[1]

    key = key.reshape([batch, kv_head, 1, seq_len, head_dim])
    key = paddle.tile(key, [1, 1, heads // kv_head, 1, 1])
    key = key.reshape([batch, heads, seq_len, head_dim])

    value = value.reshape([batch, kv_head, 1, seq_len, head_dim])
    value = paddle.tile(value, [1, 1, heads // kv_head, 1, 1])
    value = value.reshape([batch, heads, seq_len, head_dim])

    qk_res = paddle.matmul(query, key, transpose_y=True)
    attention = qk_res * scale
    attention = attention + mask
    softmax_result = paddle.nn.functional.softmax(attention, -1)
    result = paddle.matmul(softmax_result, value)
    return result


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.2",
)
class TestMemEffAttentionVariableAPI(unittest.TestCase):
    def setUp(self):
        self.name = "MemEffAPIVariable_fp32"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 1
        self.num_head = 8
        self.kv_num_head = 2
        self.seq_len = 64
        self.dim_head = 16
        self.seq_lens = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        )
        self.shape = (
            self.batch_size,
            self.num_head,
            self.seq_len,
            self.dim_head,
        )
        self.shape_kv = (
            self.batch_size,
            self.kv_num_head,
            self.seq_len,
            self.dim_head,
        )
        self.dtype = 'float32'
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])

    def test_all(self):
        paddle.disable_static()

        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key = np.random.random(self.shape_kv)
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value = np.random.random(self.shape_kv)
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        out_ = naive_attention_impl(q, k, v, self.attention_mask, self.scale)

        out = variable_length_memory_efficient_attention(
            q,
            k,
            v,
            self.seq_lens,
            self.seq_lens,
            self.attention_mask,
            self.scale,
        )

        for i in range(self.batch_size):
            np.testing.assert_allclose(
                out.numpy()[i, :, : self.seq_lens[i], :],
                out_[i, :, : self.seq_lens[i], :],
                rtol=5e-03,
                atol=1e-03,
            )


class TestMemEffAPIVariableDtypeFP16(TestMemEffAttentionVariableAPI):
    def setUp(self):
        self.name = "MemEffAPIVariable_fp16"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 3
        self.num_head = 16
        self.kv_num_head = 2
        self.seq_len = 64
        self.dim_head = 32
        self.seq_lens = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        )
        self.shape = (
            self.batch_size,
            self.num_head,
            self.seq_len,
            self.dim_head,
        )
        self.shape_kv = (
            self.batch_size,
            self.kv_num_head,
            self.seq_len,
            self.dim_head,
        )
        self.dtype = 'float16'
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "MemEffAPIVariableDtypeBF16 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestMemEffAPIVariableDtypeBF16(TestMemEffAttentionVariableAPI):
    def setUp(self):
        self.name = "MemEffAPIVariable_bf16"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.kv_num_head = 2
        self.seq_len = 32
        self.dim_head = 128
        self.seq_lens = paddle.to_tensor(
            [
                self.seq_len // 2,
                self.seq_len,
            ],
            "int32",
        )
        self.shape = (
            self.batch_size,
            self.num_head,
            self.seq_len,
            self.dim_head,
        )
        self.shape_kv = (
            self.batch_size,
            self.kv_num_head,
            self.seq_len,
            self.dim_head,
        )
        self.dtype = 'bfloat16'
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.2",
)
class TestMemEffAPIVariableDtypeFP16Static(unittest.TestCase):
    def setUp(self):
        self.name = "MemEffAPIVariableStatic_fp16"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 3
        self.num_head = 16
        self.kv_num_head = 2
        self.seq_len = 64
        self.dim_head = 32
        self.seq_lens = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        ).numpy()
        self.shape = (
            self.batch_size,
            self.num_head,
            self.seq_len,
            self.dim_head,
        )
        self.shape_kv = (
            self.batch_size,
            self.kv_num_head,
            self.seq_len,
            self.dim_head,
        )
        self.dtype = 'float16'
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        ).numpy()
        self.q = np.random.random(self.shape).astype(self.dtype)
        self.k = np.random.random(self.shape_kv).astype(self.dtype)
        self.v = np.random.random(self.shape_kv).astype(self.dtype)
        self.scale = 1.0 / np.sqrt(self.shape[-1])

        self.ref_out = naive_attention_impl(
            paddle.to_tensor(self.q),
            paddle.to_tensor(self.k),
            paddle.to_tensor(self.v),
            paddle.to_tensor(self.attention_mask),
            self.scale,
        )

    def test_all(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            q = paddle.static.data(
                name="query", shape=self.shape, dtype=self.dtype
            )
            k = paddle.static.data(
                name="key", shape=self.shape_kv, dtype=self.dtype
            )
            v = paddle.static.data(
                name="value", shape=self.shape_kv, dtype=self.dtype
            )
            mask = paddle.static.data(
                name="mask",
                shape=[self.batch_size, 1, self.seq_len, self.seq_len],
                dtype=self.dtype,
            )
            seq_lens = paddle.static.data(
                name="seq_lens", shape=[self.batch_size, 1], dtype="int32"
            )
            out = variable_length_memory_efficient_attention(
                q, k, v, seq_lens, seq_lens, mask, self.scale
            )
            exe = base.Executor()
            res = exe.run(
                feed={
                    "query": self.q,
                    "key": self.k,
                    "value": self.v,
                    "seq_lens": self.seq_lens,
                    "mask": self.attention_mask,
                },
                fetch_list=[out],
            )
        paddle.disable_static()
        np.testing.assert_allclose(res[0], self.ref_out, rtol=5e-03, atol=1e-03)


if __name__ == '__main__':
    unittest.main()
