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

import unittest
import numpy as np
import paddle
from paddle.incubate.nn.functional import (
    speculative_decoding_multihead_attention,
)

# paddle.disable_static()

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
    cache_k_dequant_scales=None,
    cache_v_dequant_scales=None,
    use_cachekv_int8="None",
):
    batch = query.shape[0]
    heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    kv_head = key.shape[1]

    key = key.reshape([batch, kv_head, 1, seq_len, head_dim])
    key = paddle.tile(key, [1, 1, heads // kv_head, 1, 1])
    key = key.reshape([batch, heads, seq_len, head_dim])

    if use_cachekv_int8 == "dynamic":
        unsqueeze_shape = [2, 3]
    elif use_cachekv_int8 == "static":
        unsqueeze_shape = [0, 2, 3]
    if pre_cache_k is not None:
        key = paddle.concat([pre_cache_k, key], axis=2)
    if cache_k is not None:
        if cache_k_dequant_scales is not None:
            dequant_cache_k = (
                (cache_k.astype('float32') - 128.0)
                * cache_k_dequant_scales.unsqueeze(unsqueeze_shape)
            ).astype(key.dtype)
            key = paddle.concat([dequant_cache_k, key], axis=2)
        else:
            key = paddle.concat([cache_k, key], axis=2)

    value = value.reshape([batch, kv_head, 1, seq_len, head_dim])
    value = paddle.tile(value, [1, 1, heads // kv_head, 1, 1])
    value = value.reshape([batch, heads, seq_len, head_dim])
    if pre_cache_v is not None:
        value = paddle.concat([pre_cache_v, value], axis=2)
    if cache_v is not None:
        if cache_v_dequant_scales is not None:
            dequant_cache_v = (
                (cache_v.astype('float32') - 128.0)
                * cache_v_dequant_scales.unsqueeze(unsqueeze_shape)
            ).astype(value.dtype)
            value = paddle.concat([dequant_cache_v, value], axis=2)
        else:
            value = paddle.concat([cache_v, value], axis=2)

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


# @unittest.skipIf(
#     not core.is_compiled_with_cuda()
#     or get_cuda_version() < 11040
#     or not is_sm_supported,
#     "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
#     "and device's compute capability must be 8.x or 90",
# )
class TestSpecuDecodingAttention(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestSpecuDecodingAttention"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
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
            self.num_head,
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

        self.tgt_mask = paddle.randn(
            [self.batch_size, self.num_head, 1, self.seq_len + 1],
            dtype=self.dtype,
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
            q, k, v, None, None, None, None, None, self.scale
        )
        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )
        print(self.cu_seqlens_q)
        print(self.cu_seqlens_k)
        print("-------------")
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
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            0, # token_num_in_cache
            64,  # max_seq_len,
            False,  # use_neox_rotary_style
        )

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )

        # decoder
        naive_cache_k, naive_cache_v = self.cache_k, self.cache_v

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        print("seq_len_decoder: ", self.seq_lens_decoder)
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            self.seq_len,
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
        print("token num: ", self.token_num)
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
        ).reshape([self.batch_size, -1])
        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, 1, self.seq_lens_this_time)
        print(self.cu_seqlens_q)
        print(self.cu_seqlens_k)

        out_ = (
            naive_attention_impl(
                q,
                k,
                v,
                naive_cache_k,
                naive_cache_v,
                None,
                None,
                self.tgt_mask,
                self.scale,
            )
            .transpose([0, 2, 1, 3])
            .reshape([self.batch_size, -1])
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
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            self.token_num,  # token_num_in_cache
            1,  # seq_len,
            False,  # use_neox_rotary_style
        )
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-02,
            atol=5e-02,
        )

if __name__ == '__main__':
    unittest.main()
