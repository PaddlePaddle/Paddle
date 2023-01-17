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

import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.nn.functional as F
from paddle import tensor
from paddle.fluid import layers
from paddle.fluid.framework import default_main_program
from paddle.incubate.nn import FusedMultiTransformer
from paddle.incubate.nn.functional import fused_multi_transformer
from paddle.nn.layer.common import Dropout, Linear
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.transformer import _convert_attention_mask

default_main_program().random_seed = 42


class TestFusedMultiTransformerOp(OpTest):
    def setUp(self):
        self.config()
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
            initializer=paddle.fluid.initializer.Constant(value=0.0005)
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
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr,
        )
        self.v_proj = Linear(
            self.vdim,
            self.embed_dim,
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

        self.x_type = np.float32
        self.attn_mask_type = np.float64
        # self.attn_mask_type = np.bool
        self.pre_layer_norm = True
        self.has_attn_mask = True

        # has_cache_kv, gen_cache_kv, stage
        # False,        False,        not generation
        # True,         True,         generation context stage
        # True,         False,        generation decoder stage
        self.has_cache_kv = False
        self.gen_cache_kv = False
        self.has_pre_cache = False

        self.training = False

        self.layers = 4

        self.batch_size = 8
        self.query_length = 128
        self.cache_length = 128
        self.pre_cache_num = 64
        self.head_dim = 64
        self.num_heads = 16
        self.embed_dim = self.head_dim * self.num_heads

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
                    self.num_heads,
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

        if self.has_pre_cache:
            out_seq_len += self.pre_cache_num
            self.pre_cache_kv = np.random.uniform(
                -1,
                1,
                (
                    2,
                    self.batch_size,
                    self.num_heads,
                    self.pre_cache_num,
                    self.head_dim,
                ),
            ).astype(self.x_type)

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
        self.key, self.value = self.query, self.query

        self.dout = np.random.uniform(
            -1, 1, (self.batch_size, self.query_length, self.embed_dim)
        ).astype(self.x_type)

    def GetBaselineOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        tensor_query = paddle.to_tensor(self.query, stop_gradient=False)

        cache_kvs = []
        cache_kv = None
        if self.has_cache_kv:
            cache_kv = paddle.to_tensor(self.cache_kv, stop_gradient=False)

        if self.has_pre_cache:
            pre_cache_kv = paddle.to_tensor(
                self.pre_cache_kv, stop_gradient=False
            )

        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None

        for i in range(self.layers):
            residual = tensor_query
            ln1_out = tensor_query
            if self.pre_layer_norm:
                ln1_out = self.norm(tensor_query)

            q = self.q_proj(ln1_out)
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q_out = tensor.transpose(x=q, perm=[0, 2, 1, 3])
            k = self.k_proj(ln1_out)
            v = self.v_proj(ln1_out)
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k_out = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v_out = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            if self.has_cache_kv:
                # [1, B, n_head, cache_seq_len, head_dim]
                cache_k, cache_v = paddle.split(cache_kv, 2)
                cache_k = paddle.squeeze(cache_k, axis=0)
                cache_v = paddle.squeeze(cache_v, axis=0)
                # [B, n_head, cache_seq_len + seq_len, head_dim]
                # out_seq_len = cache_seq_len + seq_len
                if self.debug:
                    print('q out is')
                    print(q_out[0, 0, :, :])
                    print('cache k out seq=128')
                    print(k_out[0, 0, :, :])
                if self.gen_cache_kv:
                    cache_kvs.append((k_out, v_out))
                else:
                    k_out = paddle.concat([cache_k, k_out], axis=-2)
                    v_out = paddle.concat([cache_v, v_out], axis=-2)

            if self.has_pre_cache:
                pre_cache_k, pre_cache_v = paddle.split(pre_cache_kv, 2)
                pre_cache_k = paddle.squeeze(pre_cache_k, axis=0)
                pre_cache_v = paddle.squeeze(pre_cache_v, axis=0)
                k_out = paddle.concat([pre_cache_k, k_out], axis=-2)
                v_out = paddle.concat([pre_cache_v, v_out], axis=-2)

            # [B, n_head, seq_len, head_dim] * [B, n_head, out_seq_len, head_dim]
            # --> [B, n_head, seq_len, out_seq_len]
            qk_out = paddle.matmul(x=q_out, y=k_out, transpose_y=True)
            qk_out = paddle.scale(qk_out, scale=self.head_dim**-0.5)

            if self.debug:
                print('qk out is')
                print(qk_out[0][0][0])

            if attn_mask is not None:
                attn_mask = _convert_attention_mask(attn_mask, qk_out.dtype)
                attn_mask_out = qk_out + attn_mask
                if self.debug:
                    print('attn mask out is')
                    print(attn_mask_out[0][0][0])
                softmax_out = F.softmax(attn_mask_out)
            else:
                softmax_out = F.softmax(qk_out)

            if self.debug:
                print('softmax out is')
                print(softmax_out[0][0][0])
            if self.dropout_prob:
                dropout_out = F.dropout(
                    softmax_out,
                    self.dropout_prob,
                    training=self.training,
                    mode="upscale_in_train",
                )
                # [B, n_head, seq_len, out_seq_len] * [B, n_head, out_seq_len, head_dim]
                # --> [B, n_head, seq_len, head_dim]
                qktv_out = tensor.matmul(dropout_out, v_out)
            else:
                qktv_out = tensor.matmul(softmax_out, v_out)

            fmha_out = tensor.transpose(qktv_out, perm=[0, 2, 1, 3])
            if self.debug:
                print('fmha out is')
                print(fmha_out[0][0][0])
            out_linear_in = tensor.reshape(
                x=fmha_out, shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]]
            )
            out = self.out_proj(out_linear_in)

            residual_out = residual + self.dropout(out)
            if not self.pre_layer_norm:
                attn_out = self.norm(residual_out)
            else:
                attn_out = residual_out

            ffn_ln_out = attn_out
            if self.pre_layer_norm:
                ffn_ln_out = self.ffn_norm(attn_out)

            ffn1_out = self.ffn1_proj(ffn_ln_out)
            ffn1_out = self.dropout(self.activation(ffn1_out))
            ffn2_out = self.ffn2_proj(ffn1_out)

            residual_out = attn_out + self.dropout(ffn2_out)
            final_out = residual_out
            if not self.pre_layer_norm:
                final_out = self.ffn_norm(residual_out)

            tensor_query = final_out

        if self.has_cache_kv and self.gen_cache_kv:
            return final_out, cache_kvs
        return final_out

    def GetFusedMultiTransformerOut(self):
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
            qkv_bias = np.concatenate(
                (q_proj_bias.numpy(), k_proj_bias.numpy(), v_proj_bias.numpy())
            )
            qkv_bias = qkv_bias.reshape((3, self.num_heads, self.head_dim))
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
        qkv_weight = qkv_weight.reshape(
            (3, self.num_heads, self.head_dim, self.embed_dim)
        )

        x = paddle.to_tensor(self.query, stop_gradient=False)
        cache_kvs, cache_kv = None, None
        time_step = None
        pre_caches, pre_cache = None, None
        if self.has_cache_kv:
            cache_kvs = []

            max_seq_length = (self.cache_length + 128) // 128 * 128
            cache_kv = np.zeros(
                [
                    2,
                    self.batch_size,
                    self.num_heads,
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
                    self.num_heads,
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
                    self.num_heads,
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

        if self.has_pre_cache:
            cache_kvs = []
            max_seq_length = (
                self.cache_length + 128
            ) // 128 * 128 + self.pre_cache_num
            cache_kv = np.zeros(
                [
                    2,
                    self.batch_size,
                    self.num_heads,
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
            pre_caches=pre_caches,
            time_step=time_step,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_prob,
            activation=self.act_method,
            training=self.training,
        )

        if self.has_cache_kv:
            return final_out[0], final_out[1]

        if self.has_pre_cache:
            return final_out[0]

        return final_out

    def GetFusedMultiTransformerOutStatic(self):
        paddle.enable_static()
        x = paddle.fluid.data('x', self.query.shape, self.query.dtype)
        cache_kvs, cache_kv = None, None
        time_step = None
        time_step_feed = None
        pre_caches, pre_cache = None, None
        if self.has_cache_kv:
            cache_kvs = []

            max_seq_length = (self.cache_length + 128) // 128 * 128
            cache_kv = np.zeros(
                [
                    2,
                    self.batch_size,
                    self.num_heads,
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
            cache_k_tmp = self.cache_kv[0].reshape(
                [
                    self.batch_size,
                    self.num_heads,
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
                    self.num_heads,
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
                time_step = layers.fill_constant(
                    shape=[1], dtype="int32", value=0, force_cpu=True
                )
                time_step_feed = self.cache_length

        if self.has_pre_cache:
            cache_kvs = []
            max_seq_length = (
                self.cache_length + 128
            ) // 128 * 128 + self.pre_cache_num
            cache_kv = np.zeros(
                [
                    2,
                    self.batch_size,
                    self.num_heads,
                    max_seq_length,
                    self.head_dim,
                ],
                dtype=self.x_type,
            )
            pre_caches = []

        attn_mask = None
        epsilon = 1e-05
        ln2_epsilon = 1e-05

        qkv_weights_attr, qkv_biases_attr = [], []
        out_weights_attr, out_biases_attr = [], []
        ln_scales_attr, ln_biases_attr = [], []
        ffn1_weights_attr, ffn1_biases_attr = [], []
        ffn2_weights_attr, ffn2_biases_attr = [], []
        ffn_ln_scales_attr, ffn_ln_biases_attr = [], []

        if self.has_cache_kv:
            cache_kvs_feed = []
        if self.has_pre_cache:
            cache_kvs_feed = []
            pre_caches_feed = []

        for i in range(self.layers):
            qkv_weights_attr.append(self.weight_attr)
            qkv_biases_attr.append(self.bias_attr)
            out_weights_attr.append(self.weight_attr)
            out_biases_attr.append(self.bias_attr)
            ln_scales_attr.append(self.ln_w_attr)
            ln_biases_attr.append(self.ln_b_attr)
            ffn1_weights_attr.append(self.weight_attr)
            ffn1_biases_attr.append(self.bias_attr)
            ffn2_weights_attr.append(self.weight_attr)
            ffn2_biases_attr.append(self.bias_attr)
            ffn_ln_scales_attr.append(self.ln_w_attr)
            ffn_ln_biases_attr.append(self.ln_b_attr)

        transformer = FusedMultiTransformer(
            self.embed_dim,
            self.num_heads,
            4 * self.embed_dim,
            self.dropout_prob,
            activation=self.act_method,
            normalize_before=self.pre_layer_norm,
            ln_scale_attrs=ln_scales_attr,
            ln_bias_attrs=ln_biases_attr,
            qkv_weight_attrs=qkv_weights_attr,
            qkv_bias_attrs=qkv_biases_attr,
            linear_weight_attrs=out_weights_attr,
            linear_bias_attrs=out_biases_attr,
            ffn_ln_scale_attrs=ffn_ln_scales_attr,
            ffn_ln_bias_attrs=ffn_ln_biases_attr,
            ffn1_weight_attrs=ffn1_weights_attr,
            ffn1_bias_attrs=ffn1_biases_attr,
            ffn2_weight_attrs=ffn2_weights_attr,
            ffn2_bias_attrs=ffn2_biases_attr,
        )

        transformer.eval()

        for i in range(self.layers):
            if self.has_cache_kv:
                cache_kvs.append(
                    layers.fill_constant(
                        shape=cache_kv.shape, dtype=cache_kv.dtype, value=0
                    )
                )
                cache_kvs_feed.append(cache_kv)

            if self.has_pre_cache:
                cache_kvs.append(
                    layers.fill_constant(
                        shape=cache_kv.shape, dtype=cache_kv.dtype, value=0
                    )
                )
                cache_kvs_feed.append(cache_kv)
                pre_caches.append(
                    layers.fill_constant(
                        shape=self.pre_cache_kv.shape,
                        dtype=self.pre_cache_kv.dtype,
                        value=0,
                    )
                )
                pre_caches_feed.append(self.pre_cache_kv)

        final_out = transformer(
            x,
            attn_mask=attn_mask,
            caches=cache_kvs,
            pre_caches=pre_caches,
            time_step=time_step,
        )[0]
        exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
        exe.run(paddle.static.default_startup_program())
        feed_data = {
            'x': self.query,
            'cache_kvs': cache_kvs_feed,
            'pre_caches': pre_caches_feed,
            'time_step': time_step_feed,
            'attn_mask': attn_mask,
        }
        out = exe.run(
            paddle.fluid.default_main_program(),
            feed=feed_data,
            fetch_list=[final_out],
        )
        paddle.disable_static()
        return out[0]

    def test_fused_multi_transformer_op(self):
        final_out_ref = self.GetBaselineOut()
        final_out = self.GetFusedMultiTransformerOut()
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

            if self.gen_cache_kv:
                final_out_ref, cache_kvs = final_out_ref
                for i in range(self.layers):
                    cache_k_ref = cache_kvs[i][0]
                    cache_v_ref = cache_kvs[i][1]

                    cache_k = cache_kv_out[i][0, :]
                    cache_k = cache_k.reshape(
                        [bsz, num_head, v_elems, max_seq_len, elems]
                    )
                    cache_k = cache_k[:, :, :, : self.cache_length, :]
                    cache_k = cache_k.transpose([0, 1, 3, 2, 4])
                    cache_k = cache_k.reshape(
                        [bsz, num_head, self.cache_length, head_dim]
                    )

                    cache_v = cache_kv_out[i][1, :, :, : self.cache_length, :]

                    np.testing.assert_allclose(
                        cache_k_ref, cache_k, rtol=self.rtol, atol=self.atol
                    )
                    np.testing.assert_allclose(
                        cache_v_ref, cache_v, rtol=self.rtol, atol=self.atol
                    )
                    if i == 0:
                        break

        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=self.rtol, atol=self.atol
        )


class TestFusedMultiTransformerOpFp16(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.x_type = np.float16
        self.layers = 3  # odd layers


class TestFusedMultiTransformerOpActReluFp16(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.x_type = np.float16
        self.act_method = "relu"
        self.layers = 3  # odd layers


class TestFusedMultiTransformerOpCacheKV(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.query_length = 1
        self.key_length, self.value_length = 1, 1
        self.layers = 3  # odd layers


class TestFusedMultiTransformerOpCacheKVFp16(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.query_length = 1
        self.key_length, self.value_length = 1, 1
        self.x_type = np.float16


class TestFusedMultiTransformerOpGenCacheKV(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = True


class TestFusedMultiTransformerOpGenCacheKVFp16(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = True
        self.x_type = np.float16
        self.layers = 3  # odd layers


class TestFusedMultiTransformerOpPostLayerNormFp16(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.x_type = np.float16
        self.layers = 3  # odd layers
        self.pre_layer_norm = False


class TestFusedMultiTransformerOpCacheKVPostLayerNorm(
    TestFusedMultiTransformerOp
):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.query_length = 1
        self.key_length, self.value_length = 1, 1
        self.layers = 3  # odd layers
        self.pre_layer_norm = False


class TestFusedMultiTransformerOpCacheKVPostLayerNormFp16(
    TestFusedMultiTransformerOp
):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.query_length = 1
        self.key_length, self.value_length = 1, 1
        self.x_type = np.float16
        self.pre_layer_norm = False


class TestFusedMultiTransformerOpGenCacheKVPostLayerNorm(
    TestFusedMultiTransformerOp
):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = True
        self.pre_layer_norm = False


class TestFusedMultiTransformerOpGenCacheKVPostLayerNormFp16(
    TestFusedMultiTransformerOp
):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = True
        self.x_type = np.float16
        self.layers = 3  # odd layers
        self.pre_layer_norm = False


class TestFusedMultiTransformerOpPreCache(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_pre_cache = True
        self.x_type = np.float16


class TestFusedMultiTransformerOpPreCacheStatic(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_pre_cache = True
        self.has_attn_mask = False
        self.x_type = np.float32
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.fluid.initializer.Constant(0.0)
        )
        self.bias_attr = paddle.ParamAttr(
            initializer=paddle.fluid.initializer.Constant(0.0005)
        )
        self.ln_w_attr = paddle.ParamAttr(
            initializer=paddle.fluid.initializer.Constant(1.0)
        )
        self.ln_b_attr = paddle.ParamAttr(
            initializer=paddle.fluid.initializer.Constant(0.0)
        )

    def test_fused_multi_transformer_op(self):
        final_out_ref = self.GetBaselineOut()
        final_out = self.GetFusedMultiTransformerOutStatic()

        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=self.rtol, atol=self.atol
        )


if __name__ == "__main__":
    unittest.main()
