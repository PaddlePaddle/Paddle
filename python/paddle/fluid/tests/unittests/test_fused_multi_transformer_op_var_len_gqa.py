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

# Note(@RichardWooSJTU): Since the results of MHA are consistent with those on the Python side,
# it is sufficient to compare the results of fuse_mt GQA with those of fuse_mt MHA here.

import os
import random
import unittest

import numpy as np
from eager_op_test import OpTest
from test_sparse_attention_op import get_cuda_version

import paddle
import paddle.nn.functional as F
from paddle.fluid import core
from paddle.fluid.framework import default_main_program
from paddle.incubate.nn.functional import fused_multi_transformer
from paddle.nn.layer.common import Dropout, Linear
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.transformer import _convert_attention_mask

seed = 42

random.seed(seed)
default_main_program().random_seed = seed
np.random.seed(seed)
paddle.seed(seed)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformerInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOp(OpTest):
    def setUp(self):
        self.config()
        self.kv_num_heads = (
            self.gqa_group_size if self.gqa_group_size > 0 else self.num_heads
        )

        self.generate_input_data()

        self.rtol = 1e-5
        # FIXME(wangxi): Because there is a problem with the test precision
        #  on A100, atol is temporarily set to 1e-2, and it will be
        #  changed back after the precision problem is solved.
        self.atol = 1e-2
        # make sure local development precision
        if "V100" in paddle.device.cuda.get_device_name():
            self.atol = 1e-4
        if self.x_type is np.float16:
            self.atol = 1e-1

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_multi_transformer"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = False

        bias_attr = paddle.fluid.ParamAttr(
            initializer=paddle.paddle.nn.initializer.Constant(value=0.0005)
        )
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=bias_attr,
        )
        # bias_attr=self.bias_attr)

        self.k_proj = Linear(
            self.kdim,
            self.kv_num_heads * self.head_dim,
            self.weight_attr,
            bias_attr=self.bias_attr,
        )
        self.v_proj = Linear(
            self.vdim,
            self.kv_num_heads * self.head_dim,
            self.weight_attr,
            bias_attr=self.bias_attr,
        )
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr,
        )

        self.ffn1_proj = Linear(
            self.embed_dim,
            4 * self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr,
        )
        self.ffn2_proj = Linear(
            4 * self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr,
        )

        paddle.set_default_dtype(np.float32)
        self.norm = LayerNorm(self.embed_dim)
        self.ffn_norm = LayerNorm(self.embed_dim)

        paddle.set_default_dtype(self.x_type)
        self.dropout = Dropout(self.dropout_prob, mode="upscale_in_train")
        self.activation = getattr(F, self.act_method)

    def config(self):
        # for debug
        self.debug = False
        # self.debug = True

        self.x_type = np.float32
        self.attn_mask_type = np.float64
        # self.attn_mask_type = np.bool_
        self.pre_layer_norm = True
        self.has_attn_mask = True

        # has_cache_kv, gen_cache_kv, stage
        # False,        False,        not generation
        # True,         True,         generation context stage
        # True,         False,        generation decoder stage
        self.has_cache_kv = False
        self.gen_cache_kv = False
        self.has_pre_cache = False
        self.rotary_embs = None
        self.rotary_emb_dims = 0
        self.neox_rotary_style = False

        self.remove_padding = False

        self.training = False

        self.layers = 4

        self.batch_size = 8
        self.query_length = 128
        self.cache_length = 128
        self.pre_cache_num = 64
        self.head_dim = 64
        self.num_heads = 16
        self.embed_dim = self.head_dim * self.num_heads

        # For GQA
        self.gqa_group_size = 8
        self.use_fake_mha = False

        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        self.act_method = 'gelu'
        self.weight_attr = None
        self.bias_attr = None
        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = (
            self.query_length,
            self.query_length,
        )

    def generate_input_data(self):
        self.query = np.random.uniform(
            -1, 1, (self.batch_size, self.query_length, self.embed_dim)
        ).astype(self.x_type)

        out_seq_len = self.key_length

        if self.has_cache_kv:
            assert self.training is False, ValueError(
                'cache_kv can only used in inference'
            )
            self.cache_kv = np.random.uniform(
                -1,
                1,
                (
                    2,
                    self.batch_size,
                    self.kv_num_heads,
                    self.cache_length,
                    self.head_dim,
                ),
            ).astype(self.x_type)

            if self.gen_cache_kv:
                self.cache_kv[:] = 0
            else:
                out_seq_len += self.cache_length
        else:
            self.cache_kv = None

        if self.remove_padding:
            if self.has_cache_kv and not self.gen_cache_kv:
                # decoder
                self.seq_lens = [
                    random.randint(1, self.cache_length)
                    for _ in range(self.batch_size)
                ]
                self.seq_lens[
                    random.randint(0, self.batch_size)
                ] = self.cache_length
                self.seq_lens = np.array(self.seq_lens).astype(np.int32)
            else:
                self.seq_lens = [
                    random.randint(1, self.query_length)
                    for _ in range(self.batch_size)
                ]
                self.seq_lens[
                    random.randint(0, self.batch_size)
                ] = self.query_length
                self.seq_lens = np.array(self.seq_lens).astype(np.int32)

        if self.has_attn_mask:
            # [B, n_head, seq_len, out_seq_len]
            self.attn_mask = np.ones(
                (self.batch_size, 1, self.query_length, out_seq_len),
                dtype=self.attn_mask_type,
            )
            if self.attn_mask_type == np.int64:
                self.attn_mask = np.tril(self.attn_mask)
            elif self.attn_mask_type == np.float64:
                if self.has_cache_kv and not self.gen_cache_kv:
                    # NOTE: decoder stage, -1(out_seq_len) should no mask
                    self.attn_mask[:, :, :, -2] = 0.0
                    self.attn_mask = (self.attn_mask - 1.0) * 1e4
                else:
                    self.attn_mask = (np.tril(self.attn_mask) - 1.0) * 1e4
            elif self.attn_mask_type == np.bool_:
                if self.has_cache_kv and not self.gen_cache_kv:
                    self.attn_mask[:, :, :, -2] = 0
                else:
                    self.attn_mask = np.tril(self.attn_mask)
            else:
                raise ValueError(
                    "'attn_mask_type' should be 'int64' or 'float64'."
                )
        else:
            self.attn_mask = None

        if self.rotary_emb_dims > 0:
            self.rotary_emb_dims = (
                1 if not self.neox_rotary_style else self.rotary_emb_dims
            )
            self.rotary_emb = np.random.uniform(
                -1,
                1,
                (
                    2,
                    self.batch_size,
                    1,
                    self.query_length,
                    self.head_dim // 2 // self.rotary_emb_dims,
                ),
            ).astype('float32')
            if self.neox_rotary_style:
                concat_nums = 2 * self.rotary_emb_dims
                rotary_embs = []
                for _ in range(concat_nums):
                    rotary_embs.append(self.rotary_emb)
                self.rotary_embs = np.concatenate(rotary_embs, -1)
            else:
                rotary_emb = paddle.to_tensor(self.rotary_emb)
                self.rotary_embs = paddle.reshape(
                    paddle.stack([rotary_emb, rotary_emb], axis=-1),
                    [2, self.batch_size, 1, self.query_length, self.head_dim],
                ).numpy()

        self.key, self.value = self.query, self.query

        self.dout = np.random.uniform(
            -1, 1, (self.batch_size, self.query_length, self.embed_dim)
        ).astype(self.x_type)

    def GetFusedMultiTransformerOut(self):
        if self.has_cache_kv and self.use_fake_mha:
            if self.gen_cache_kv:
                self.cache_kv[:] = 0
            shape = [
                2,
                self.batch_size,
                self.num_heads,
                self.cache_length,
                self.head_dim,
            ]
            self.cache_kv = paddle.to_tensor(self.cache_kv)
            self.cache_kv = paddle.stack(
                [self.cache_kv] * (self.num_heads // self.kv_num_heads), axis=3
            )

            # import pdb;pdb.set_trace()

            self.cache_kv = paddle.reshape(self.cache_kv, shape).numpy()

        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_proj_weight = paddle.to_tensor(
            self.q_proj.weight, stop_gradient=False
        )
        k_proj_weight = paddle.to_tensor(
            self.k_proj.weight, stop_gradient=False
        )
        v_proj_weight = paddle.to_tensor(
            self.v_proj.weight, stop_gradient=False
        )

        if self.use_fake_mha:
            origin_shape = [self.embed_dim, self.embed_dim]

            k_proj_weight = paddle.reshape(
                k_proj_weight,
                [self.embed_dim, self.kv_num_heads, self.head_dim],
            )
            v_proj_weight = paddle.reshape(
                v_proj_weight,
                [self.embed_dim, self.kv_num_heads, self.head_dim],
            )

            k_proj_weight = paddle.stack(
                [k_proj_weight] * (self.num_heads // self.kv_num_heads), axis=-2
            )
            v_proj_weight = paddle.stack(
                [v_proj_weight] * (self.num_heads // self.kv_num_heads), axis=-2
            )

            k_proj_weight = paddle.reshape(k_proj_weight, origin_shape)
            v_proj_weight = paddle.reshape(v_proj_weight, origin_shape)

        out_linear_weight = paddle.to_tensor(
            self.out_proj.weight, stop_gradient=False
        )
        ffn1_weight = paddle.to_tensor(
            self.ffn1_proj.weight, stop_gradient=False
        )
        ffn2_weight = paddle.to_tensor(
            self.ffn2_proj.weight, stop_gradient=False
        )

        if self.bias_attr is False:
            qkv_bias_tensor = None
            out_linear_bias = None
        else:
            q_proj_bias = paddle.to_tensor(
                self.q_proj.bias, stop_gradient=False
            )
            k_proj_bias = paddle.to_tensor(
                self.k_proj.bias, stop_gradient=False
            )
            v_proj_bias = paddle.to_tensor(
                self.v_proj.bias, stop_gradient=False
            )
            if self.use_fake_mha:
                origin_shape = [self.embed_dim]
                k_proj_bias = paddle.reshape(
                    k_proj_bias, [self.kv_num_heads, self.head_dim]
                )
                v_proj_bias = paddle.reshape(
                    v_proj_bias, [self.kv_num_heads, self.head_dim]
                )

                k_proj_bias = paddle.reshape(
                    paddle.stack(
                        [k_proj_bias] * (self.num_heads // self.kv_num_heads),
                        axis=-2,
                    ),
                    origin_shape,
                )
                v_proj_bias = paddle.reshape(
                    paddle.stack(
                        [v_proj_bias] * (self.num_heads // self.kv_num_heads),
                        axis=-2,
                    ),
                    origin_shape,
                )

            qkv_bias = np.concatenate(
                (q_proj_bias.numpy(), k_proj_bias.numpy(), v_proj_bias.numpy())
            )
            if self.gqa_group_size <= 0 or self.use_fake_mha:
                qkv_bias = qkv_bias.reshape((3, self.num_heads, self.head_dim))
            else:
                qkv_bias = qkv_bias.reshape(
                    (self.num_heads + 2 * self.kv_num_heads, self.head_dim)
                )
            qkv_bias_tensor = paddle.to_tensor(qkv_bias, stop_gradient=False)
            out_linear_bias = paddle.to_tensor(
                self.out_proj.bias, stop_gradient=False
            )
            ffn1_bias = paddle.to_tensor(
                self.ffn1_proj.bias, stop_gradient=False
            )
            ffn2_bias = paddle.to_tensor(
                self.ffn2_proj.bias, stop_gradient=False
            )

        ln_scale = paddle.to_tensor(self.norm.weight, stop_gradient=False)
        ln_bias = paddle.to_tensor(self.norm.bias, stop_gradient=False)
        ffn_ln_scale = paddle.to_tensor(
            self.ffn_norm.weight, stop_gradient=False
        )
        ffn_ln_bias = paddle.to_tensor(self.ffn_norm.bias, stop_gradient=False)

        q_proj_weight = q_proj_weight.numpy().transpose((1, 0))
        k_proj_weight = k_proj_weight.numpy().transpose((1, 0))
        v_proj_weight = v_proj_weight.numpy().transpose((1, 0))
        qkv_weight = np.concatenate(
            (q_proj_weight, k_proj_weight, v_proj_weight)
        )
        if self.gqa_group_size <= 0 or self.use_fake_mha:
            qkv_weight = qkv_weight.reshape(
                (3, self.num_heads, self.head_dim, self.embed_dim)
            )
        else:
            qkv_weight = qkv_weight.reshape(
                (
                    self.num_heads + 2 * self.kv_num_heads,
                    self.head_dim,
                    self.embed_dim,
                )
            )

        if self.rotary_emb_dims > 0:
            rotary_embs = paddle.to_tensor(
                self.rotary_embs, stop_gradient=False
            )
        else:
            rotary_embs = None

        x = paddle.to_tensor(self.query, stop_gradient=False)
        cache_kvs, cache_kv = None, None
        time_step = None
        pre_caches = None

        fuse_kv_num_heads = (
            self.kv_num_heads
            if self.gqa_group_size > 0 and not self.use_fake_mha
            else self.num_heads
        )
        if self.has_cache_kv:
            cache_kvs = []

            max_seq_length = (self.cache_length + 128) // 128 * 128
            cache_kv = np.zeros(
                [
                    2,
                    self.batch_size,
                    fuse_kv_num_heads,
                    max_seq_length,
                    self.head_dim,
                ],
                dtype=self.x_type,
            )

            elems = 4
            if self.x_type is np.float16:
                elems = 8

            assert self.head_dim % elems == 0
            v_elems = self.head_dim // elems

            # [B, num_head, 128, head_dim]
            # cache_k_tmp = self.cache_kv[0, :]
            # [B, num_head, 128, head_dim / 4, 4]
            cache_k_tmp = self.cache_kv[0].reshape(
                [
                    self.batch_size,
                    fuse_kv_num_heads,
                    self.cache_length,
                    v_elems,
                    elems,
                ]
            )
            # [B, num_head, head_dim / 4, 128, 4]
            cache_k_tmp = cache_k_tmp.transpose([0, 1, 3, 2, 4])

            cache_kv[0, :].reshape(
                [
                    self.batch_size,
                    fuse_kv_num_heads,
                    v_elems,
                    max_seq_length,
                    elems,
                ]
            )[:, :, :, : self.cache_length, :] = cache_k_tmp

            cache_kv[1, :, :, : self.cache_length, :] = self.cache_kv[1]
            if self.gen_cache_kv:
                assert self.query_length == self.cache_length
                cache_kv[:] = 0
            else:
                time_step = paddle.to_tensor(
                    [self.cache_length], dtype='int32', place=paddle.CPUPlace()
                )

        if self.remove_padding:
            seq_lens = paddle.to_tensor(self.seq_lens, dtype='int32')
        else:
            seq_lens = None

        if self.has_pre_cache:
            cache_kvs = []
            max_seq_length = (
                self.cache_length + 128
            ) // 128 * 128 + self.pre_cache_num
            cache_kv = np.zeros(
                [
                    2,
                    self.batch_size,
                    fuse_kv_num_heads,
                    max_seq_length,
                    self.head_dim,
                ],
                dtype=self.x_type,
            )
            pre_caches = []

        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None
        qkv_weight_tensor = paddle.to_tensor(qkv_weight, stop_gradient=False)
        epsilon = 1e-05
        ln2_epsilon = 1e-05

        if attn_mask is not None and self.attn_mask_type != np.bool_:
            attn_mask = _convert_attention_mask(attn_mask, x.dtype)

        qkv_weights, qkv_biases = [], []
        out_weights, out_biases = [], []
        ln_scales, ln_biases = [], []
        ffn1_weights, ffn1_biases = [], []
        ffn2_weights, ffn2_biases = [], []
        ffn_ln_scales, ffn_ln_biases = [], []
        for i in range(self.layers):
            qkv_weights.append(qkv_weight_tensor)
            qkv_biases.append(qkv_bias_tensor)
            out_weights.append(out_linear_weight)
            out_biases.append(out_linear_bias)
            ln_scales.append(ln_scale)
            ln_biases.append(ln_bias)
            ffn1_weights.append(ffn1_weight)
            ffn1_biases.append(ffn1_bias)
            ffn2_weights.append(ffn2_weight)
            ffn2_biases.append(ffn2_bias)
            ffn_ln_scales.append(ffn_ln_scale)
            ffn_ln_biases.append(ffn_ln_bias)
            if self.has_cache_kv:
                cache_kvs.append(
                    paddle.to_tensor(cache_kv, stop_gradient=False)
                )
            if self.has_pre_cache:
                cache_kvs.append(
                    paddle.to_tensor(cache_kv, stop_gradient=False)
                )
                pre_caches.append(
                    paddle.to_tensor(self.pre_cache_kv, stop_gradient=False)
                )

        final_out = fused_multi_transformer(
            x,
            ln_scales,
            ln_biases,
            qkv_weights,
            qkv_biases,
            out_weights,
            out_biases,
            ffn_ln_scales,
            ffn_ln_biases,
            ffn1_weights,
            ffn1_biases,
            ffn2_weights,
            ffn2_biases,
            pre_layer_norm=self.pre_layer_norm,
            epsilon=epsilon,
            cache_kvs=cache_kvs,
            rotary_embs=rotary_embs,
            rotary_emb_dims=self.rotary_emb_dims,
            pre_caches=pre_caches,
            time_step=time_step,
            seq_lens=seq_lens,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_prob,
            activation=self.act_method,
            training=self.training,
            use_neox_rotary_style=self.neox_rotary_style,
            gqa_group_size=self.gqa_group_size if not self.use_fake_mha else -1,
        )

        if self.has_cache_kv:
            return final_out[0], final_out[1]

        if self.has_pre_cache:
            return final_out[0]

        return final_out

    def test_fused_multi_transformer_op(self):
        if not self.remove_padding:
            return

        final_out = self.GetFusedMultiTransformerOut()
        self.use_fake_mha = True
        final_out_ref = self.GetFusedMultiTransformerOut()

        if self.has_cache_kv:
            final_out, cache_kv_out = final_out
            s = cache_kv_out[0].shape
            bsz = s[1]
            num_head = s[2]
            max_seq_len = s[3]
            head_dim = s[4]
            elems = 8 if self.x_type is np.float16 else 4
            v_elems = head_dim // elems

            if self.debug:
                print("cache_k out timestep=128")
                print(
                    cache_kv_out[0].reshape(
                        [2, bsz, num_head, v_elems, max_seq_len, elems]
                    )[0, 0, 0, :, self.cache_length, :]
                )

                print("cache_v out timestep=128")
                print(cache_kv_out[0][1, 0, 0, self.cache_length, :])

            if self.remove_padding and not self.gen_cache_kv:
                # test decoder
                final_out_ref, cache_kvs = final_out_ref
                for i in range(self.batch_size):
                    for j in range(self.layers):
                        cache_k = cache_kv_out[j][0, :, -1]
                        cache_v = cache_kv_out[j][1, :, -1]

                        cache_k_ref = cache_kvs[j][0, :, -1]
                        cache_v_ref = cache_kvs[j][1, :, -1]
                        np.testing.assert_allclose(
                            cache_k_ref,
                            cache_k,
                            rtol=self.rtol,
                            atol=self.atol,
                        )
                        np.testing.assert_allclose(
                            cache_v_ref,
                            cache_v,
                            rtol=self.rtol,
                            atol=self.atol,
                        )

            if self.gen_cache_kv:
                final_out_ref, cache_kvs = final_out_ref
                for i in range(self.layers):
                    cache_k_ref = cache_kvs[i][0, :, 0]
                    cache_v_ref = cache_kvs[i][1, :, 0]

                    cache_k = cache_kv_out[i][0, :, 0]
                    cache_v = cache_kv_out[i][1, :, 0]

                    if self.remove_padding:
                        for i in range(self.batch_size):
                            np.testing.assert_allclose(
                                cache_k_ref,
                                cache_k,
                                rtol=self.rtol,
                                atol=self.atol,
                            )
                            np.testing.assert_allclose(
                                cache_v_ref,
                                cache_v,
                                rtol=self.rtol,
                                atol=self.atol,
                            )
                    else:
                        np.testing.assert_allclose(
                            cache_k_ref, cache_k, rtol=self.rtol, atol=self.atol
                        )
                        np.testing.assert_allclose(
                            cache_v_ref, cache_v, rtol=self.rtol, atol=self.atol
                        )
                    if i == 0:
                        break

        if self.remove_padding:
            for i in range(self.batch_size):
                np.testing.assert_allclose(
                    final_out_ref[i, : self.seq_lens[i]],
                    final_out[i, : self.seq_lens[i]],
                    rtol=self.rtol,
                    atol=self.atol,
                )
        else:
            np.testing.assert_allclose(
                final_out_ref, final_out, rtol=self.rtol, atol=self.atol
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformerInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOpVariableGenCache1(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = True
        self.remove_padding = True
        self.x_type = np.float16
        self.layers = 3  # odd layers
        # self.pre_layer_norm = False


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformerInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOpVariableGenCache2(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = True
        self.remove_padding = True
        self.layers = 4  # even layers
        if (
            "FLAGS_fmha_mode" in os.environ
            and os.environ["FLAGS_fmha_mode"] == "flash_attention_v2"
        ):
            self.x_type = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformerInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOpVariableGenCache3(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = True
        self.remove_padding = True
        self.layers = 4  # even layers
        self.rotary_emb_dims = 2

        if (
            "FLAGS_fmha_mode" in os.environ
            and os.environ["FLAGS_fmha_mode"] == "flash_attention_v2"
        ):
            self.x_type = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformerInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOpVariableGenCache4(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = True
        self.remove_padding = True
        self.layers = 3  # odd layers
        self.rotary_emb_dims = 2
        if (
            "FLAGS_fmha_mode" in os.environ
            and os.environ["FLAGS_fmha_mode"] == "flash_attention_v2"
        ):
            self.x_type = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformerInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOpVariableDecoder1(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = False
        self.remove_padding = True
        self.query_length = 1
        self.key_length, self.value_length = 1, 1
        self.x_type = np.float16
        self.layers = 3  # odd layers
        self.pre_layer_norm = False


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformerInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOpVariableDecoder2(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = False
        self.remove_padding = True
        self.query_length = 1
        self.key_length, self.value_length = 1, 1
        self.layers = 4  # even layers
        if (
            "FLAGS_fmha_mode" in os.environ
            and os.environ["FLAGS_fmha_mode"] == "flash_attention_v2"
        ):
            self.x_type = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformerInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOpVariableDecoder3(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = False
        self.remove_padding = True
        self.query_length = 1
        self.key_length, self.value_length = 1, 1
        self.layers = 4  # even layers
        self.rotary_emb_dims = 2
        if (
            "FLAGS_fmha_mode" in os.environ
            and os.environ["FLAGS_fmha_mode"] == "flash_attention_v2"
        ):
            self.x_type = np.float16


if __name__ == "__main__":
    unittest.main()
