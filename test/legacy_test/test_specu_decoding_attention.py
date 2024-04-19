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
from test_block_multihead_attention import RopeEmbedding

import paddle
from paddle.framework import core
from paddle.incubate.nn.functional.speculative_decoding_multihead_attention import (
    speculative_decoding_multihead_attention,
)

paddle.seed(2024)
np.random.seed(2024)


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


def naive_attention_impl(
    query,
    key,
    value,
    cache_k=None,
    cache_v=None,
    pre_cache_k=None,
    pre_cache_v=None,
    mask=None,
    scale=1.0,
):
    """
    query: [batch, heads, seq_len, head_dim]
    key: [batch, heads, seq_len, head_dim]
    value: [batch, heads, seq_len, head_dim]
    cache_k: [batch, seq_len, heads, head_dim]
    cache_v: [batch, seq_len, heads, head_dim]
    """
    batch = query.shape[0]
    heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    kv_head = key.shape[1]

    key = key.reshape([batch, kv_head, 1, seq_len, head_dim])
    key = paddle.tile(key, [1, 1, heads // kv_head, 1, 1])
    key = key.reshape([batch, heads, seq_len, head_dim])

    if pre_cache_k is not None:
        key = paddle.concat([pre_cache_k, key], axis=2)
    if cache_k is not None:
        cache_k_transed = paddle.transpose(cache_k, [0, 2, 1, 3])
        key = paddle.concat([cache_k_transed, key], axis=2)

    value = value.reshape([batch, kv_head, 1, seq_len, head_dim])
    value = paddle.tile(value, [1, 1, heads // kv_head, 1, 1])
    value = value.reshape([batch, heads, seq_len, head_dim])
    if pre_cache_v is not None:
        value = paddle.concat([pre_cache_v, value], axis=2)
    if cache_v is not None:
        cache_v_transed = paddle.transpose(cache_v, [0, 2, 1, 3])
        value = paddle.concat([cache_v_transed, value], axis=2)

    qk_res = paddle.matmul(query, key, transpose_y=True)
    attention = qk_res * scale
    if mask is not None:
        attention = attention + mask
    softmax_result = paddle.nn.functional.softmax(attention, -1)
    result = paddle.matmul(softmax_result, value)
    return result


def remove_padding(seq_lens, cu_seq_lens, inputs, token_num):
    bsz, num_head, seq_len, dim_head = inputs.shape
    output = paddle.zeros(
        shape=[token_num, num_head * dim_head], dtype=inputs.dtype
    )
    inputs = inputs.transpose([0, 2, 1, 3]).reshape([bsz, seq_len, -1])
    for i in range(bsz):
        seq_len_now = seq_lens[i]
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        output[start_idx:end_idx, :] = inputs[i, :seq_len_now, :]
    return output


def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time)
    cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = paddle.sum(seq_lens_this_time)
    padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i]
        cum_offset = cum_offsets[i]
        for j in range(seq_len_now):
            padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
        cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return padding_offsets, cum_offsets[:-1], cu_seqlens_q, cu_seqlens_k


def create_attn_mask(
    mask_type,
    batch_size,
    seq_lens,
    pre_cache_length=0,
):
    max_seq_len = max(seq_lens)
    mask = paddle.zeros(
        [batch_size, 1, max_seq_len, max_seq_len + pre_cache_length],
        dtype=mask_type,
    )
    mask[:, :, :, :pre_cache_length] = 1
    for i in range(batch_size):
        seq_len = seq_lens[i]
        mask[i, 0, :seq_len, :seq_len] = (
            paddle.tril(paddle.ones(shape=(seq_len, seq_len), dtype=mask_type))
            - 1
        ) * 1e4
    return mask


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestSpecuDecodingAttention(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestSpecuDecodingAttention"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 1
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.seq_lens_encoder = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        )
        self.seq_lens_decoder = paddle.to_tensor(
            [
                0,
            ]
            * self.batch_size,
            "int32",
        )
        self.seq_lens_this_time = self.seq_lens_encoder
        self.shape = (
            self.batch_size,
            self.num_head,
            self.seq_len,
            self.dim_head,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])

        self.cache_shape = (  # (bsz, num_head, cache_seq_len, dim_head)
            self.batch_size,
            self.seq_len,
            self.num_head,
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

        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(
            self.batch_size, self.seq_len, self.seq_lens_this_time
        )

        self.token_num = self.padding_offset.shape[0]

    def test_all(self):
        paddle.disable_static()
        # encoder
        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key = np.random.random(self.shape)
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value = np.random.random(self.shape)
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        qkv = paddle.stack(
            [
                q.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num, self.hid_dim]
                ),
            ],
            axis=1,
        ).reshape([self.token_num, -1])
        out_ = naive_attention_impl(
            q, k, v, None, None, None, None, self.attention_mask, self.scale
        )
        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )
        out, qkv_out, _, _ = speculative_decoding_multihead_attention(
            qkv,
            self.cache_k,
            self.cache_v,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            None,  # rope
            None,  # attn_mask
            None,  # qkv_bias
            self.seq_len,  # max_enc_len_this_time
            0,  # max_dec_len_this_time
            0,  # num_tokens_in_cache
            self.max_dec_len,  # max_seq_len,
            False,  # use_neox_rotary_style
        )

        k_out = qkv_out[:, self.hid_dim : 2 * self.hid_dim]
        k_out = paddle.reshape(
            k_out, [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        )
        self.cache_k[:, : self.seq_len, :, :] = k_out

        v_out = qkv_out[:, 2 * self.hid_dim :]
        v_out = paddle.reshape(
            v_out, [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        )
        self.cache_v[:, : self.seq_len, :, :] = v_out

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )

        # decoder
        naive_cache_k, naive_cache_v = self.cache_k, self.cache_v

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.token_num_decoder_phase = 5
        self.seq_lens_this_time[:] = self.token_num_decoder_phase

        self.shape = (
            self.batch_size,
            self.num_head,
            self.token_num_decoder_phase,  # token_num > 1 in decoder phase
            self.dim_head,
        )
        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key = np.random.random(self.shape)
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value = np.random.random(self.shape)
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        qkv = paddle.stack(
            [
                q.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num_decoder_phase, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num_decoder_phase, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num_decoder_phase, self.hid_dim]
                ),
            ],
            axis=1,
        ).reshape([self.token_num_decoder_phase, -1])
        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(
            self.batch_size,
            self.token_num_decoder_phase,
            self.seq_lens_this_time,
        )

        out_ = (
            naive_attention_impl(
                q,
                k,
                v,
                naive_cache_k,
                naive_cache_v,
                None,
                None,
                None,
                self.scale,
            )
            .transpose([0, 2, 1, 3])
            .reshape([self.token_num_decoder_phase, -1])
        )

        # Because we use flash-attn in decoder phase, so we put the whole key and value into flash-attn.
        self.cu_seqlens_k[1] += self.token_num
        max_dec_len_this_time = self.seq_len + self.token_num_decoder_phase
        (
            out,
            qkv_out,
            cache_k_out,
            cache_v_out,
        ) = speculative_decoding_multihead_attention(
            qkv,
            self.cache_k,
            self.cache_v,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            None,  # rope
            None,  # attn_mask
            None,  # qkv_bias
            0,  # max_enc_len_this_time
            max_dec_len_this_time,  # max_dec_len_this_time
            self.token_num,  # num_tokens_in_cache,
            self.max_dec_len,  # max_seq_len,
            False,  # use_neox_rotary_style
        )

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-2,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            cache_k_out.numpy(),
            self.cache_k.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            cache_v_out.numpy(),
            self.cache_v.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )


class TestSpecuDecodingAttnRoPE(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestSpecuDecodingAttention"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 1
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.rope = RopeEmbedding()
        self.seq_lens_encoder = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        )
        self.seq_lens_decoder = paddle.to_tensor(
            [
                0,
            ]
            * self.batch_size,
            "int32",
        )
        self.seq_lens_this_time = self.seq_lens_encoder
        self.shape = (
            self.batch_size,
            self.num_head,
            self.seq_len,
            self.dim_head,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])

        self.cache_shape = (  # (bsz, num_head, cache_seq_len, dim_head)
            self.batch_size,
            self.seq_len,
            self.num_head,
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

        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(
            self.batch_size, self.seq_len, self.seq_lens_this_time
        )

        self.token_num = self.padding_offset.shape[0]

    def get_rotary_position_embedding(self, position_ids, head_dim):
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros(
            (2, bsz, max_seq_len, 1, head_dim // 2), dtype="float32"
        )
        inv_freq = 10000 ** (
            -paddle.arange(0, head_dim, 2, dtype="float32") / head_dim
        )

        # shape: [B, S, D/2]
        freqs = paddle.einsum(
            "ij,k->ijk", position_ids.cast("float32"), inv_freq
        )
        # shape: [B, S, D]
        # emb = paddle.stack([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, head_dim))
        emb = paddle.stack([freqs], axis=-1).reshape(
            (bsz, max_seq_len, head_dim // 2)
        )
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

    def test_all(self):
        paddle.disable_static()
        tmp_position_ids = paddle.arange(
            self.seq_len + self.max_dec_len
        ).reshape((1, -1))

        self.rope_emb = self.get_rotary_position_embedding(
            tmp_position_ids, self.dim_head
        )
        # encoder
        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key = np.random.random(self.shape)
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value = np.random.random(self.shape)
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        qkv = paddle.stack(
            [
                q.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num, self.hid_dim]
                ),
            ],
            axis=1,
        ).reshape([self.token_num, -1])
        sinusoidal_pos = self.rope._rotary_position_embedding(
            self.seq_len, self.dim_head, self.dtype
        )
        q, k = self.rope._apply_rope(sinusoidal_pos, q, k)

        out_ = naive_attention_impl(
            q, k, v, None, None, None, None, self.attention_mask, self.scale
        )

        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )

        out, qkv_out, _, _ = speculative_decoding_multihead_attention(
            qkv,
            self.cache_k,
            self.cache_v,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.rope_emb,  # rope
            None,  # attn_mask
            None,  # qkv_bias
            self.seq_len,  # max_enc_len_this_time
            0,  # max_dec_len_this_time
            0,  # num_tokens_in_cache
            self.max_dec_len,  # max_seq_len,
            False,  # use_neox_rotary_style
        )

        k_out = qkv_out[:, self.hid_dim : 2 * self.hid_dim]
        k_out = paddle.reshape(
            k_out, [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        )
        self.cache_k[:, : self.seq_len, :, :] = k_out

        v_out = qkv_out[:, 2 * self.hid_dim :]
        v_out = paddle.reshape(
            v_out, [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        )
        self.cache_v[:, : self.seq_len, :, :] = v_out

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=1e-04,
            atol=5e-04,
        )

        # decoder
        naive_cache_k, naive_cache_v = self.cache_k, self.cache_v
        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.token_num_decoder_phase = 5
        self.seq_lens_this_time[:] = self.token_num_decoder_phase

        self.shape = (
            self.batch_size,
            self.num_head,
            self.token_num_decoder_phase,  # multi token
            self.dim_head,
        )
        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key = np.random.random(self.shape)
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value = np.random.random(self.shape)
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        qkv = paddle.stack(
            [
                q.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num_decoder_phase, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num_decoder_phase, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.token_num_decoder_phase, self.hid_dim]
                ),
            ],
            axis=1,
        ).reshape([self.token_num_decoder_phase, -1])
        sinusoidal_pos = self.rope._rotary_position_embedding(
            self.seq_len + self.token_num_decoder_phase,
            self.dim_head,
            self.dtype,
        )[:, :, -self.token_num_decoder_phase :, :]
        q, k = self.rope._apply_rope(sinusoidal_pos, q, k)

        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(
            self.batch_size,
            self.token_num_decoder_phase,
            self.seq_lens_this_time,
        )

        out_ = (
            naive_attention_impl(
                q,
                k,
                v,
                naive_cache_k,
                naive_cache_v,
                None,
                None,
                None,
                self.scale,
            )
            .transpose([0, 2, 1, 3])
            .reshape([self.token_num_decoder_phase, -1])
        )
        self.cu_seqlens_k[1] += self.token_num
        max_dec_len_this_time = self.seq_len + self.token_num_decoder_phase
        (
            out,
            qkv_out,
            cache_k_out,
            cache_v_out,
        ) = speculative_decoding_multihead_attention(
            qkv,
            self.cache_k,
            self.cache_v,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.rope_emb,  # rope
            None,  # attn_mask
            None,  # qkv_bias
            0,  # max_enc_len_this_time
            max_dec_len_this_time,  # max_dec_len_this_time
            self.token_num,  # num_tokens_in_cache,
            self.max_dec_len,  # max_seq_len,
            False,  # use_neox_rotary_style
        )

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-2,
            atol=3e-2,
        )
        np.testing.assert_allclose(
            cache_k_out.numpy(),
            self.cache_k.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            cache_v_out.numpy(),
            self.cache_v.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )


if __name__ == '__main__':
    unittest.main()
