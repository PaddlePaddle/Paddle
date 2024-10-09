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
from paddle.incubate.nn.functional import block_multihead_attention
from paddle.static import Program, program_guard

paddle.seed(2023)
np.random.seed(2023)


is_sm8x = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] >= 0
)

is_sm9x = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 9
    and paddle.device.cuda.get_device_capability()[1] >= 0
)

is_sm7x = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 7
    and paddle.device.cuda.get_device_capability()[1] >= 0
)

is_sm_supported = is_sm8x or is_sm9x or is_sm7x


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


class RopeEmbedding:
    def _rotary_position_embedding(self, seq_len, head_dim, dtype):
        pos_seq = paddle.arange(0, seq_len, 1, dtype=dtype)
        indices = paddle.arange(0, head_dim, 2, dtype=dtype)
        indices = 1 / 10000 ** (indices / head_dim)

        sinusoid_inp = pos_seq.unsqueeze(1) * indices.unsqueeze(0)
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1
        )
        pos_emb = paddle.reshape(pos_emb, (1, 1, seq_len, head_dim))
        pos_emb.stop_gradient = True
        return pos_emb

    def _apply_rope(self, rp, q, k, v=None):
        # sin [sequence_length, embed_size_per_head//2]
        # cos [sequence_length, embed_size_per_head//2]
        sin, cos = paddle.chunk(rp, 2, axis=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), rp.shape)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), rp.shape)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q, cos_pos), paddle.multiply(rotate_half_q, sin_pos)
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k, cos_pos), paddle.multiply(rotate_half_k, sin_pos)
        )
        if v is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_v = paddle.reshape(
                paddle.stack([-v[:, :, :, 1::2], v[:, :, :, 0::2]], axis=-1),
                paddle.shape(v),
            )
            value = paddle.add(
                paddle.multiply(v, cos_pos),
                paddle.multiply(rotate_half_v, sin_pos),
            )
            return query, key, value
        return query, key


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


def block_cache_to_naive_cache(
    cache_k, cache_v, bsz, block_tables, cache_seq_len
):
    _, num_head, blocksize, dim_head = cache_k.shape
    out_cache_k = paddle.zeros(
        shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_k.dtype
    )
    out_cache_v = paddle.zeros(
        shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_v.dtype
    )
    for i in range(bsz):
        for j in range(cache_seq_len):
            out_cache_k[i, :, j, :] = cache_k[
                block_tables[i, j // blocksize], :, j % blocksize, :
            ]
            out_cache_v[i, :, j, :] = cache_v[
                block_tables[i, j // blocksize], :, j % blocksize, :
            ]
    return out_cache_k, out_cache_v


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnEncDec(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnEncDec"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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

        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style,
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
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
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
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
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            self.tgt_mask,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-02,
            atol=5e-02,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnEncDecSkipGetMaxLen(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnEncDecSkipGetMaxLen"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.max_enc_len_this_time = paddle.to_tensor(
            [self.seq_len], "int32"
        ).cpu()
        self.max_dec_len_this_time = paddle.to_tensor([0], "int32").cpu()
        self.shape = (
            self.batch_size,
            self.num_head,
            self.seq_len,
            self.dim_head,
        )
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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

        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            self.max_enc_len_this_time,  # max_enc_len_this_time
            self.max_dec_len_this_time,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style,
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.max_enc_len_this_time = paddle.to_tensor([0], "int32").cpu()
        self.max_dec_len_this_time = paddle.to_tensor(
            [self.seq_len], "int32"
        ).cpu()
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
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
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
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
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            self.max_enc_len_this_time,  # max_enc_len_this_time
            self.max_dec_len_this_time,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            self.tgt_mask,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-02,
            atol=5e-02,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnRoPE(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnRoPE"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.rope = RopeEmbedding()
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            self.rope_emb,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
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
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
            ],
            axis=1,
        ).reshape([self.batch_size, -1])

        sinusoidal_pos = self.rope._rotary_position_embedding(
            self.seq_len + 1, self.dim_head, self.dtype
        )[:, :, -1:, :]
        q, k = self.rope._apply_rope(sinusoidal_pos, q, k)

        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, 1, self.seq_lens_this_time)

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
            .reshape([self.batch_size, -1])
        )
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            self.rope_emb,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-02,
            atol=5e-02,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnPreCache(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnPreCacbe"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.pre_cache_length = 64
        self.max_seq_len = self.seq_len + self.pre_cache_length
        self.block_num_per_seq = (
            self.max_seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
            self.dim_head,
        )
        self.pre_cache_shape = (
            self.batch_size,
            self.num_head,
            self.pre_cache_length,
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
            self.pre_cache_length,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.pre_cache_k = paddle.randn(
            shape=self.pre_cache_shape, dtype=self.dtype
        )
        self.pre_cache_v = paddle.randn(
            shape=self.pre_cache_shape, dtype=self.dtype
        )
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
            q,
            k,
            v,
            None,
            None,
            self.pre_cache_k,
            self.pre_cache_v,
            self.attention_mask,
            self.scale,
        )
        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )
        out = block_multihead_attention(
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
            self.block_tables,
            self.pre_cache_k,  # pre_key_cache
            self.pre_cache_v,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            self.attention_mask,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len + self.pre_cache_length,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
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
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
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
            .reshape([self.batch_size, -1])
        )
        out = block_multihead_attention(
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
            self.block_tables,
            self.pre_cache_k,  # pre_key_cache
            self.pre_cache_v,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            self.attention_mask,  # attn_mask
            None,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=5e-02,
            atol=5e-02,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnEncStatic(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnEncStatic"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
        out_ = naive_attention_impl(
            q, k, v, None, None, None, None, self.attention_mask, self.scale
        )
        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )

        qkv_numpy = (
            paddle.stack(
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
            )
            .reshape([self.token_num, -1])
            .numpy()
        )
        paddle.enable_static()
        with program_guard(Program(), Program()):
            qkv = paddle.static.data(
                name="qkv",
                shape=(self.token_num, 3 * self.hid_dim),
                dtype=self.dtype,
            )
            cache_k = paddle.static.data(
                name="cache_k", shape=self.cache_shape, dtype=self.dtype
            )
            cache_v = paddle.static.data(
                name="cache_v", shape=self.cache_shape, dtype=self.dtype
            )
            seq_lens_encoder = paddle.static.data(
                name="seq_lens_encoder", shape=(self.batch_size,), dtype='int32'
            )
            seq_lens_decoder = paddle.static.data(
                name="seq_lens_decoder", shape=(self.batch_size,), dtype='int32'
            )
            seq_lens_this_time = paddle.static.data(
                name="seq_lens_this_time",
                shape=(self.batch_size,),
                dtype='int32',
            )
            cu_seqlens_q = paddle.static.data(
                name="cu_seqlens_q", shape=(self.batch_size + 1,), dtype='int32'
            )
            cu_seqlens_k = paddle.static.data(
                name="cu_seqlens_k", shape=(self.batch_size + 1,), dtype='int32'
            )
            padding_offsets = paddle.static.data(
                name="padding_offset", shape=(self.token_num,), dtype='int32'
            )
            cum_offsets = paddle.static.data(
                name="cum_offset", shape=(self.batch_size,), dtype='int32'
            )
            block_tables = paddle.static.data(
                name="block_tables",
                shape=(self.batch_size, self.block_num_per_seq),
                dtype='int32',
            )

            out = block_multihead_attention(
                qkv,
                cache_k,
                cache_v,
                seq_lens_encoder,
                seq_lens_decoder,
                seq_lens_this_time,
                padding_offsets,
                cum_offsets,
                cu_seqlens_q,
                cu_seqlens_k,
                block_tables,
                None,  # pre_key_cache
                None,  # pre_value_cache
                None,  # cache_k_quant_scales
                None,  # cache_v_quant_scales
                None,  # cache_k_dequant_scales
                None,  # cache_v_dequant_scales
                None,  # qkv_out_scale
                None,  # qkv_bias
                None,  # out_shift
                None,  # out_smooth
                None,  # max_enc_len_this_time
                None,  # max_dec_len_this_time
                None,  # rotary_embs
                None,  # attn_mask
                None,  # tgt_mask
                self.seq_len,
                self.blocksize,
                False,  # use_neox_rotary_style
            )
            exe = base.Executor()
            res = exe.run(
                feed={
                    "qkv": qkv_numpy,
                    "cache_k": self.cache_k.numpy(),
                    "cache_v": self.cache_v.numpy(),
                    "seq_lens_encoder": self.seq_lens_encoder.numpy(),
                    "seq_lens_decoder": self.seq_lens_decoder.numpy(),
                    "seq_lens_this_time": self.seq_lens_this_time.numpy(),
                    "cu_seqlens_q": self.cu_seqlens_q.numpy(),
                    "cu_seqlens_k": self.cu_seqlens_k.numpy(),
                    "padding_offset": self.padding_offset.numpy(),
                    "cum_offset": self.cum_offset.numpy(),
                    "block_tables": self.block_tables.numpy(),
                },
                fetch_list=[out],
            )
        paddle.disable_static()
        np.testing.assert_allclose(
            res[0],
            out_.numpy(),
            rtol=5e-03,
            atol=1e-03,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnEncDecPTQDequant(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnEncDec"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
        query = np.random.randint(-65535, 65535, self.shape, 'int32')
        q = paddle.to_tensor(
            query, place=self.place, dtype='int32', stop_gradient=False
        )
        key = np.random.randint(-65535, 65535, self.shape, 'int32')
        k = paddle.to_tensor(
            key, place=self.place, dtype='int32', stop_gradient=False
        )
        value = np.random.randint(-65535, 65535, self.shape, 'int32')
        v = paddle.to_tensor(
            value, place=self.place, dtype='int32', stop_gradient=False
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

        q = q.transpose([0, 2, 1, 3]).reshape([self.token_num, self.hid_dim])
        k = k.transpose([0, 2, 1, 3]).reshape([self.token_num, self.hid_dim])
        v = v.transpose([0, 2, 1, 3]).reshape([self.token_num, self.hid_dim])

        q_out_scale = 10.0 / paddle.max(q, axis=0).astype('float32')
        k_out_scale = 10.0 / paddle.max(k, axis=0).astype('float32')
        v_out_scale = 10.0 / paddle.max(v, axis=0).astype('float32')

        qkv_out_scale = paddle.concat(
            [q_out_scale, k_out_scale, v_out_scale], axis=0
        )

        q_bias = paddle.ones([self.hid_dim], dtype=self.dtype)
        k_bias = paddle.ones([self.hid_dim], dtype=self.dtype)
        v_bias = paddle.ones([self.hid_dim], dtype=self.dtype)

        qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=-1)

        # dequant
        q = (q.astype('float32') * q_out_scale).astype(self.dtype)
        k = (k.astype('float32') * k_out_scale).astype(self.dtype)
        v = (v.astype('float32') * v_out_scale).astype(self.dtype)

        # add bias
        q = q + q_bias
        k = k + k_bias
        v = v + v_bias

        # transpose to origin
        q = q.reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])
        k = k.reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])
        v = v.reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])

        out_ = naive_attention_impl(
            q, k, v, None, None, None, None, self.attention_mask, self.scale
        )
        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            qkv_out_scale,  # qkv_out_scale
            qkv_bias,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style,
            compute_dtype="fp16",
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=100,
            atol=1,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
            self.dim_head,
        )
        query = np.random.randint(-65535, 65535, self.shape, 'int32')
        q = paddle.to_tensor(
            query, place=self.place, dtype='int32', stop_gradient=False
        )
        key = np.random.randint(-65535, 65535, self.shape, 'int32')
        k = paddle.to_tensor(
            key, place=self.place, dtype='int32', stop_gradient=False
        )
        value = np.random.randint(-65535, 65535, self.shape, 'int32')
        v = paddle.to_tensor(
            value, place=self.place, dtype='int32', stop_gradient=False
        )

        qkv = paddle.stack(
            [
                q.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
            ],
            axis=1,
        ).reshape([self.batch_size, -1])

        q = q.transpose([0, 2, 1, 3]).reshape([self.batch_size, self.hid_dim])
        k = k.transpose([0, 2, 1, 3]).reshape([self.batch_size, self.hid_dim])
        v = v.transpose([0, 2, 1, 3]).reshape([self.batch_size, self.hid_dim])

        q_out_scale = 1.0 / paddle.max(q, axis=0).astype('float32')
        k_out_scale = 1.0 / paddle.max(k, axis=0).astype('float32')
        v_out_scale = 1.0 / paddle.max(v, axis=0).astype('float32')

        qkv_out_scale = paddle.concat(
            [q_out_scale, k_out_scale, v_out_scale], axis=0
        )

        q_bias = paddle.ones([self.hid_dim], dtype=self.dtype) * 0.1
        k_bias = paddle.ones([self.hid_dim], dtype=self.dtype) * 0.1
        v_bias = paddle.ones([self.hid_dim], dtype=self.dtype) * 0.1

        qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=-1)

        # dequant
        q = (q.astype('float32') * q_out_scale).astype(self.dtype)
        k = (k.astype('float32') * k_out_scale).astype(self.dtype)
        v = (v.astype('float32') * v_out_scale).astype(self.dtype)

        # add bias
        q = q + q_bias
        k = k + k_bias
        v = v + v_bias

        # transpose to origin
        q = q.reshape(
            [self.batch_size, 1, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])
        k = k.reshape(
            [self.batch_size, 1, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])
        v = v.reshape(
            [self.batch_size, 1, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])

        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, 1, self.seq_lens_this_time)

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
            .reshape([self.batch_size, -1])
        )
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            qkv_out_scale,  # qkv_out_scale
            qkv_bias,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
            compute_dtype="fp16",
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=100,
            atol=1,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnEncDecPTQDequantQuantShiftSmooth(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnEncDec"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
        query = np.random.randint(-65535, 65535, self.shape, 'int32')
        q = paddle.to_tensor(
            query, place=self.place, dtype='int32', stop_gradient=False
        )
        key = np.random.randint(-65535, 65535, self.shape, 'int32')
        k = paddle.to_tensor(
            key, place=self.place, dtype='int32', stop_gradient=False
        )
        value = np.random.randint(-65535, 65535, self.shape, 'int32')
        v = paddle.to_tensor(
            value, place=self.place, dtype='int32', stop_gradient=False
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

        q = q.transpose([0, 2, 1, 3]).reshape([self.token_num, self.hid_dim])
        k = k.transpose([0, 2, 1, 3]).reshape([self.token_num, self.hid_dim])
        v = v.transpose([0, 2, 1, 3]).reshape([self.token_num, self.hid_dim])

        q_out_scale = 1.0 / paddle.max(q, axis=0).astype('float32')
        k_out_scale = 1.0 / paddle.max(k, axis=0).astype('float32')
        v_out_scale = 1.0 / paddle.max(v, axis=0).astype('float32')

        qkv_out_scale = paddle.concat(
            [q_out_scale, k_out_scale, v_out_scale], axis=0
        )

        q_bias = paddle.ones([self.hid_dim], dtype=self.dtype)
        k_bias = paddle.ones([self.hid_dim], dtype=self.dtype)
        v_bias = paddle.ones([self.hid_dim], dtype=self.dtype)

        qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=-1)

        # dequant
        q = (q.astype('float32') * q_out_scale).astype(self.dtype)
        k = (k.astype('float32') * k_out_scale).astype(self.dtype)
        v = (v.astype('float32') * v_out_scale).astype(self.dtype)

        # add bias
        q = q + q_bias
        k = k + k_bias
        v = v + v_bias

        # transpose to origin
        q = q.reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])
        k = k.reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])
        v = v.reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])

        out_ = naive_attention_impl(
            q, k, v, None, None, None, None, self.attention_mask, self.scale
        )

        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )

        # shift smooth
        shift = np.random.random([self.num_head * self.dim_head])
        shift = paddle.to_tensor(shift, dtype=self.dtype, place=self.place)

        smooth = np.random.random([self.num_head * self.dim_head])
        smooth = paddle.to_tensor(smooth, dtype=self.dtype, place=self.place)

        out_ = (out_ + shift) * smooth

        # quant
        out_ *= 127.0

        out_ = paddle.where(out_ <= -127, paddle.full_like(out_, -127), out_)
        out_ = paddle.where(out_ >= 127, paddle.full_like(out_, 127), out_)
        out_ = paddle.round(out_).astype('int8')

        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            qkv_out_scale,  # qkv_out_scale
            qkv_bias,  # qkv_bias
            shift,  # out_shift
            smooth,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style,
            compute_dtype="fp16",
            out_scale=1.0,
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=1,
            atol=1,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
            self.dim_head,
        )
        query = np.random.randint(-65535, 65535, self.shape, 'int32')
        q = paddle.to_tensor(
            query, place=self.place, dtype='int32', stop_gradient=False
        )
        key = np.random.randint(-65535, 65535, self.shape, 'int32')
        k = paddle.to_tensor(
            key, place=self.place, dtype='int32', stop_gradient=False
        )
        value = np.random.randint(-65535, 65535, self.shape, 'int32')
        v = paddle.to_tensor(
            value, place=self.place, dtype='int32', stop_gradient=False
        )

        qkv = paddle.stack(
            [
                q.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
            ],
            axis=1,
        ).reshape([self.batch_size, -1])

        q = q.transpose([0, 2, 1, 3]).reshape([self.batch_size, self.hid_dim])
        k = k.transpose([0, 2, 1, 3]).reshape([self.batch_size, self.hid_dim])
        v = v.transpose([0, 2, 1, 3]).reshape([self.batch_size, self.hid_dim])

        q_out_scale = 1.0 / paddle.max(q, axis=0).astype('float32')
        k_out_scale = 1.0 / paddle.max(k, axis=0).astype('float32')
        v_out_scale = 1.0 / paddle.max(v, axis=0).astype('float32')

        qkv_out_scale = paddle.concat(
            [q_out_scale, k_out_scale, v_out_scale], axis=0
        )

        q_bias = paddle.ones([self.hid_dim], dtype=self.dtype) * 0.1
        k_bias = paddle.ones([self.hid_dim], dtype=self.dtype) * 0.1
        v_bias = paddle.ones([self.hid_dim], dtype=self.dtype) * 0.1

        qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=-1)

        # dequant
        q = (q.astype('float32') * q_out_scale).astype(self.dtype)
        k = (k.astype('float32') * k_out_scale).astype(self.dtype)
        v = (v.astype('float32') * v_out_scale).astype(self.dtype)

        # add bias
        q = q + q_bias
        k = k + k_bias
        v = v + v_bias

        # transpose to origin
        q = q.reshape(
            [self.batch_size, 1, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])
        k = k.reshape(
            [self.batch_size, 1, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])
        v = v.reshape(
            [self.batch_size, 1, self.num_head, self.dim_head]
        ).transpose([0, 2, 1, 3])

        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, 1, self.seq_lens_this_time)

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
            .reshape([self.batch_size, -1])
        )

        # shift smooth
        shift = np.random.random([self.num_head * self.dim_head])
        shift = paddle.to_tensor(shift, dtype=self.dtype, place=self.place)

        smooth = np.random.random([self.num_head * self.dim_head])
        smooth = paddle.to_tensor(smooth, dtype=self.dtype, place=self.place)

        out_ = (out_ + shift) * smooth

        # quant
        out_ *= 127.0

        out_ = paddle.where(out_ <= -127, paddle.full_like(out_, -127), out_)
        out_ = paddle.where(out_ >= 127, paddle.full_like(out_, 127), out_)
        out_ = paddle.round(out_).astype('int8')

        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            qkv_out_scale,  # qkv_out_scale
            qkv_bias,  # qkv_bias
            shift,  # out_shift
            smooth,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
            compute_dtype="fp16",
            out_scale=1.0,
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=20,
            atol=57,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnEncDecQuant(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnEncDec"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
        # quant
        out_ *= 127.0

        out_ = paddle.where(out_ <= -127, paddle.full_like(out_, -127), out_)
        out_ = paddle.where(out_ >= 127, paddle.full_like(out_, 127), out_)
        out_ = paddle.round(out_).astype('int8')

        out_ = remove_padding(
            self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num
        )
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style,
            out_scale=1.0,
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=0.1,
            atol=1,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
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
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
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
            .reshape([self.batch_size, -1])
        )
        # quant
        out_ *= 127.0

        out_ = paddle.where(out_ <= -127, paddle.full_like(out_, -127), out_)
        out_ = paddle.where(out_ >= 127, paddle.full_like(out_, 127), out_)
        out_ = paddle.round(out_).astype('int8')

        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
            out_scale=1.0,
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=0.1,
            atol=1,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnEncDecCacheKVDynamicQuant(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnEncDec"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype='uint8')
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype='uint8')
        self.cache_k_quant_scales = paddle.zeros(
            shape=[self.batch_size, self.num_head], dtype='float32'
        )
        self.cache_v_quant_scales = paddle.zeros(
            shape=[self.batch_size, self.num_head], dtype='float32'
        )
        self.cache_k_dequant_scales = paddle.zeros(
            shape=[self.batch_size, self.num_head], dtype='float32'
        )
        self.cache_v_dequant_scales = paddle.zeros(
            shape=[self.batch_size, self.num_head], dtype='float32'
        )

        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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
        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            self.cache_k_quant_scales,  # cache_k_quant_scales
            self.cache_v_quant_scales,  # cache_v_quant_scales
            self.cache_k_dequant_scales,  # cache_k_dequant_scales
            self.cache_v_dequant_scales,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style,
            use_dynamic_cachekv_quant=True,
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=0.1,
            atol=1,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
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
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
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
                cache_k_dequant_scales=self.cache_k_dequant_scales,
                cache_v_dequant_scales=self.cache_v_dequant_scales,
                use_cachekv_int8="dynamic",
            )
            .transpose([0, 2, 1, 3])
            .reshape([self.batch_size, -1])
        )
        # quant

        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            self.cache_k_quant_scales,  # cache_k_quant_scales
            self.cache_v_quant_scales,  # cache_v_quant_scales
            self.cache_k_dequant_scales,  # cache_k_dequant_scales
            self.cache_v_dequant_scales,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
            use_dynamic_cachekv_quant=True,
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=0.1,
            atol=1,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11040
    or not is_sm_supported,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestBlockMultiHeadAttnEncDecCacheKVStaticQuant(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestBlockMultiHeadAttnEncDec"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 64
        self.hid_dim = self.num_head * self.dim_head
        self.blocksize = 64
        self.block_num_per_seq = (
            self.seq_len + self.max_dec_len + self.blocksize - 1
        ) // self.blocksize
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
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
        self.cache_shape = (
            self.max_block_num,
            self.num_head,
            self.blocksize,
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
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype='uint8')
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype='uint8')
        self.cache_k_quant_scales = paddle.zeros(
            shape=[self.num_head], dtype='float32'
        )
        self.cache_v_quant_scales = paddle.zeros(
            shape=[self.num_head], dtype='float32'
        )
        self.cache_k_dequant_scales = paddle.zeros(
            shape=[self.num_head], dtype='float32'
        )
        self.cache_v_dequant_scales = paddle.zeros(
            shape=[self.num_head], dtype='float32'
        )

        self.block_tables = paddle.zeros(
            shape=(self.batch_size, self.block_num_per_seq), dtype="int32"
        )
        for i in range(self.batch_size):
            need_block_num = (
                self.seq_len + self.max_dec_len + self.blocksize - 1
            ) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
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

        self.cache_k_quant_scales = (
            127.0 / paddle.max(k, axis=[0, 2, 3])
        ).astype("float32")
        self.cache_v_quant_scales = (
            127.0 / paddle.max(k, axis=[0, 2, 3])
        ).astype("float32")

        self.cache_k_dequant_scales = 1.0 / self.cache_k_quant_scales
        self.cache_v_dequant_scales = 1.0 / self.cache_v_quant_scales

        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            self.cache_k_quant_scales,  # cache_k_quant_scales
            self.cache_v_quant_scales,  # cache_v_quant_scales
            self.cache_k_dequant_scales,  # cache_k_dequant_scales
            self.cache_v_dequant_scales,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            self.seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style,
            use_dynamic_cachekv_quant=False,
        )[0]

        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=0.1,
            atol=1,
        )

        # decoder
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )

        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.shape = (
            self.batch_size,
            self.num_head,
            1,
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
                    [self.batch_size, self.hid_dim]
                ),
                k.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
                ),
                v.transpose([0, 2, 1, 3]).reshape(
                    [self.batch_size, self.hid_dim]
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
                cache_k_dequant_scales=self.cache_k_dequant_scales,
                cache_v_dequant_scales=self.cache_v_dequant_scales,
                use_cachekv_int8="static",
            )
            .transpose([0, 2, 1, 3])
            .reshape([self.batch_size, -1])
        )

        out = block_multihead_attention(
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
            self.block_tables,
            None,  # pre_key_cache
            None,  # pre_value_cache
            self.cache_k_quant_scales,  # cache_k_quant_scales
            self.cache_v_quant_scales,  # cache_v_quant_scales
            self.cache_k_dequant_scales,  # cache_k_dequant_scales
            self.cache_v_dequant_scales,  # cache_v_dequant_scales
            None,  # qkv_out_scale
            None,  # qkv_bias
            None,  # out_shift
            None,  # out_smooth
            None,  # max_enc_len_this_time
            None,  # max_dec_len_this_time
            None,  # rotary_embs
            None,  # attn_mask
            None,  # tgt_mask
            1,  # seq_len,
            self.blocksize,
            False,  # use_neox_rotary_style
            use_dynamic_cachekv_quant=False,
        )[0]
        # NOTE: The diff of decoder is a little big
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=0.1,
            atol=1,
        )


if __name__ == '__main__':
    unittest.main()
