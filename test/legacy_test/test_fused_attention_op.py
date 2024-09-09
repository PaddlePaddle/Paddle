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
import paddle.incubate.nn.functional as incubate_f
import paddle.nn.functional as F
from paddle import tensor
from paddle.nn.layer.common import Dropout, Linear
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.transformer import _convert_attention_mask

paddle.seed(42)


class TestFusedAttentionOp(OpTest):
    def setUp(self):
        self.config()
        self.generate_input_data()

        self.rtol = 1e-5
        # FIXME(limin29): Because there is a problem with the test precision
        #  on A100, atol is temporarily set to 1e-2, and it will be
        #  changed back after the precision problem is solved.
        self.atol = 1e-2
        # make sure local development precision
        if "V100" in paddle.device.cuda.get_device_name():
            self.atol = 1e-4
        if self.x_type is np.float16:
            self.atol = 1e-1

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_attention"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = True
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr,
        )
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
        paddle.set_default_dtype(np.float32)
        self.norm1 = LayerNorm(self.embed_dim)
        self.norm2 = LayerNorm(self.embed_dim)
        paddle.set_default_dtype(self.x_type)
        self.dropout = Dropout(self.dropout_prob, mode="upscale_in_train")

    def config(self):
        self.x_type = np.float32
        self.attn_mask_type = np.float64
        self.pre_layer_norm = False
        self.has_attn_mask = True
        self.has_cache_kv = False
        self.training = True

        self.batch_size = 8
        self.query_length = 128
        self.cache_length = 128
        self.head_dim = 64
        self.num_heads = 16
        self.embed_dim = self.head_dim * self.num_heads

        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        self.weight_attr = None
        self.bias_attr = None
        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = (
            self.query_length,
            self.query_length,
        )
        self.transpose_qkv_wb = False

    def generate_input_data(self):
        self.query = np.random.rand(
            self.batch_size, self.query_length, self.embed_dim
        ).astype(self.x_type)
        out_seq_len = self.key_length
        if self.has_cache_kv:
            assert self.training is False, ValueError(
                'cache_kv can only used in inference'
            )
            self.cache_kv = np.random.rand(
                2,
                self.batch_size,
                self.num_heads,
                self.cache_length,
                self.head_dim,
            ).astype(self.x_type)
            out_seq_len += self.cache_length
        else:
            self.cache_kv = None

        if self.has_attn_mask:
            # [B, n_head, seq_len, out_seq_len]
            self.attn_mask = np.ones(
                (
                    self.batch_size,
                    self.num_heads,
                    self.query_length,
                    out_seq_len,
                ),
                dtype=self.attn_mask_type,
            )
            if self.attn_mask_type == np.int64:
                self.attn_mask = np.tril(self.attn_mask)
            elif self.attn_mask_type == np.float64:
                self.attn_mask = (np.tril(self.attn_mask) - 1.0) * 1e9
            else:
                raise ValueError(
                    "'attn_mask_type' should be 'int64' or 'float64'."
                )
        else:
            self.attn_mask = None
        self.key, self.value = self.query, self.query

        self.dout = np.random.random(
            (self.batch_size, self.query_length, self.embed_dim)
        ).astype(self.x_type)

    def GetBaselineOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        tensor_query = paddle.to_tensor(self.query, stop_gradient=False)

        cache_kv = None
        if self.has_cache_kv:
            cache_kv = paddle.to_tensor(self.cache_kv, stop_gradient=False)

        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None
        residual = tensor_query

        ln1_out = tensor_query
        if self.pre_layer_norm:
            ln1_out = self.norm1(tensor_query)

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
            k_out = paddle.concat([cache_k, k_out], axis=-2)
            v_out = paddle.concat([cache_v, v_out], axis=-2)

        # [B, n_head, seq_len, head_dim] * [B, n_head, out_seq_len, head_dim]
        # --> [B, n_head, seq_len, out_seq_len]
        qk_out = paddle.matmul(x=q_out, y=k_out, transpose_y=True)
        qk_out = paddle.scale(qk_out, scale=self.head_dim**-0.5)

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, qk_out.dtype)
            attn_mask_out = qk_out + attn_mask
            softmax_out = F.softmax(attn_mask_out)
        else:
            softmax_out = F.softmax(qk_out)

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
        out_linear_in = tensor.reshape(
            x=fmha_out, shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]]
        )
        out = self.out_proj(out_linear_in)

        residual_out = residual + self.dropout(out)
        if not self.pre_layer_norm:
            final_out = self.norm1(residual_out)
        else:
            final_out = residual_out

        if self.has_cache_kv:
            return final_out

        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True
        )
        return final_out, tensor_query.grad

    def GetFusedAttentionOut(self):
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
            if not self.transpose_qkv_wb:
                qkv_bias = qkv_bias.reshape((3, self.num_heads, self.head_dim))
            qkv_bias_tensor = paddle.to_tensor(qkv_bias, stop_gradient=False)
            out_linear_bias = paddle.to_tensor(
                self.out_proj.bias, stop_gradient=False
            )

        ln1_scale = paddle.to_tensor(self.norm1.weight, stop_gradient=False)
        ln1_bias = paddle.to_tensor(self.norm1.bias, stop_gradient=False)
        ln2_scale = paddle.to_tensor(self.norm2.weight, stop_gradient=False)
        ln2_bias = paddle.to_tensor(self.norm2.bias, stop_gradient=False)

        if not self.transpose_qkv_wb:
            q_proj_weight = q_proj_weight.numpy().transpose((1, 0))
            k_proj_weight = k_proj_weight.numpy().transpose((1, 0))
            v_proj_weight = v_proj_weight.numpy().transpose((1, 0))
        else:
            q_proj_weight = q_proj_weight.numpy()
            k_proj_weight = k_proj_weight.numpy()
            v_proj_weight = v_proj_weight.numpy()

        concatenate_axis = 1 if self.transpose_qkv_wb else 0
        qkv_weight = np.concatenate(
            (q_proj_weight, k_proj_weight, v_proj_weight), axis=concatenate_axis
        )
        if not self.transpose_qkv_wb:
            qkv_weight = qkv_weight.reshape(
                (3, self.num_heads, self.head_dim, self.embed_dim)
            )

        x = paddle.to_tensor(self.query, stop_gradient=False)
        cache_kv = None
        if self.has_cache_kv:
            cache_kv = paddle.to_tensor(self.cache_kv, stop_gradient=False)
        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None
        qkv_weight_tensor = paddle.to_tensor(qkv_weight, stop_gradient=False)
        epsilon = 1e-05
        ln2_epsilon = 1e-05

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, x.dtype)
        final_out = incubate_f.fused_multi_head_attention(
            x,
            qkv_weight_tensor,
            out_linear_weight,
            self.pre_layer_norm,
            ln1_scale,
            ln1_bias,
            ln2_scale,
            ln2_bias,
            epsilon,
            qkv_bias_tensor,
            out_linear_bias,
            cache_kv,
            attn_mask,
            self.dropout_prob,
            self.attn_dropout_prob,
            ln2_epsilon,
            num_heads=self.num_heads,
            transpose_qkv_wb=self.transpose_qkv_wb,
        )

        if self.has_cache_kv:
            return final_out[0], final_out[1]

        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True
        )
        return final_out, x.grad

    def test_fused_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=self.rtol, atol=self.atol
        )
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=self.rtol, atol=self.atol
        )


class TestFusedAttentionOpBiasIsNone(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.bias_attr = False


class TestFusedAttentionAPITransposeWAndB(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.transpose_qkv_wb = True


class TestFusedAttentionAPITransposeWAndBWithoutBias(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.transpose_qkv_wb = True
        self.bias_attr = False


class TestFusedAttentionOpPreLn(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.pre_layer_norm = True


class TestFusedAttentionOpNoneAttnMask(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.pre_layer_norm = True
        self.has_attn_mask = False


class TestFusedAttentionOpFp16(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.x_type = np.float16

    def test_fused_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=self.rtol, atol=self.atol
        )
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=self.rtol, atol=self.atol
        )


class TestFusedAttentionOpCacheKV(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.training = False
        self.query_length = 1
        self.key_length, self.value_length = 1, 1

    def test_fused_attention_op(self):
        with paddle.no_grad():
            final_out_ref = self.GetBaselineOut()
            final_out, cache_kv_out = self.GetFusedAttentionOut()
            np.testing.assert_allclose(
                final_out_ref, final_out.numpy(), rtol=self.rtol, atol=self.atol
            )


class TestFusedAttentionOpParamStopGradient(OpTest):
    def setUp(self):
        self.config()
        self.generate_input_data()

        self.rtol = 1e-5
        # FIXME(limin29): Because there is a problem with the test precision
        #  on A100, atol is temporarily set to 1e-2, and it will be
        #  changed back after the precision problem is solved.
        self.atol = 1e-2
        # make sure local development precision
        if "V100" in paddle.device.cuda.get_device_name():
            self.atol = 1e-4
        if self.x_type is np.float16:
            self.atol = 1e-1

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_attention"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = True
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr,
        )
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
        paddle.set_default_dtype(np.float32)
        self.norm1 = LayerNorm(self.embed_dim)
        self.norm2 = LayerNorm(self.embed_dim)
        paddle.set_default_dtype(self.x_type)
        self.dropout = Dropout(self.dropout_prob, mode="upscale_in_train")

    def config(self):
        self.x_type = np.float32
        self.attn_mask_type = np.float64
        self.pre_layer_norm = False
        self.has_attn_mask = True
        self.has_cache_kv = False
        self.training = True

        self.batch_size = 8
        self.query_length = 128
        self.cache_length = 128
        self.head_dim = 64
        self.num_heads = 16
        self.embed_dim = self.head_dim * self.num_heads

        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        self.weight_attr = None
        self.bias_attr = None
        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = (
            self.query_length,
            self.query_length,
        )

    def generate_input_data(self):
        self.query = np.random.rand(
            self.batch_size, self.query_length, self.embed_dim
        ).astype(self.x_type)
        out_seq_len = self.key_length
        if self.has_cache_kv:
            assert self.training is False, ValueError(
                'cache_kv can only used in inference'
            )
            self.cache_kv = np.random.rand(
                2,
                self.batch_size,
                self.num_heads,
                self.cache_length,
                self.head_dim,
            ).astype(self.x_type)
            out_seq_len += self.cache_length
        else:
            self.cache_kv = None

        if self.has_attn_mask:
            # [B, n_head, seq_len, out_seq_len]
            self.attn_mask = np.ones(
                (
                    self.batch_size,
                    self.num_heads,
                    self.query_length,
                    out_seq_len,
                ),
                dtype=self.attn_mask_type,
            )
            if self.attn_mask_type == np.int64:
                self.attn_mask = np.tril(self.attn_mask)
            elif self.attn_mask_type == np.float64:
                self.attn_mask = (np.tril(self.attn_mask) - 1.0) * 1e9
            else:
                raise ValueError(
                    "'attn_mask_type' should be 'int64' or 'float64'."
                )
        else:
            self.attn_mask = None
        self.key, self.value = self.query, self.query

        self.dout = np.random.random(
            (self.batch_size, self.query_length, self.embed_dim)
        ).astype(self.x_type)

    def GetBaselineOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        tensor_query = paddle.to_tensor(self.query, stop_gradient=False)

        cache_kv = None
        if self.has_cache_kv:
            cache_kv = paddle.to_tensor(self.cache_kv, stop_gradient=False)

        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None
        residual = tensor_query

        ln1_out = tensor_query
        if self.pre_layer_norm:
            ln1_out = self.norm1(tensor_query)

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
            k_out = paddle.concat([cache_k, k_out], axis=-2)
            v_out = paddle.concat([cache_v, v_out], axis=-2)

        # [B, n_head, seq_len, head_dim] * [B, n_head, out_seq_len, head_dim]
        # --> [B, n_head, seq_len, out_seq_len]
        qk_out = paddle.matmul(x=q_out, y=k_out, transpose_y=True)
        qk_out = paddle.scale(qk_out, scale=self.head_dim**-0.5)

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, qk_out.dtype)
            attn_mask_out = qk_out + attn_mask
            softmax_out = F.softmax(attn_mask_out)
        else:
            softmax_out = F.softmax(qk_out)

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
        out_linear_in = tensor.reshape(
            x=fmha_out, shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]]
        )
        out = self.out_proj(out_linear_in)

        residual_out = residual + self.dropout(out)
        if not self.pre_layer_norm:
            final_out = self.norm1(residual_out)
        else:
            final_out = residual_out

        if self.has_cache_kv:
            return final_out

        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True
        )
        return final_out, tensor_query.grad

    def GetFusedAttentionOut(self):
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

        ln1_scale = paddle.to_tensor(self.norm1.weight, stop_gradient=False)
        ln1_bias = paddle.to_tensor(self.norm1.bias, stop_gradient=False)
        ln2_scale = paddle.to_tensor(self.norm2.weight, stop_gradient=False)
        ln2_bias = paddle.to_tensor(self.norm2.bias, stop_gradient=False)

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
        cache_kv = None
        if self.has_cache_kv:
            cache_kv = paddle.to_tensor(self.cache_kv, stop_gradient=False)
        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None
        qkv_weight_tensor = paddle.to_tensor(qkv_weight, stop_gradient=False)
        epsilon = 1e-05
        ln2_epsilon = 1e-05

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, x.dtype)
        qkv_weight_tensor.stop_gradient = True
        out_linear_weight.stop_gradient = True
        ln1_scale.stop_gradient = True
        ln1_bias.stop_gradient = True
        ln2_scale.stop_gradient = True
        ln2_bias.stop_gradient = True
        qkv_bias_tensor.stop_gradient = True
        out_linear_bias.stop_gradient = True
        final_out = incubate_f.fused_multi_head_attention(
            x,
            qkv_weight_tensor,
            out_linear_weight,
            self.pre_layer_norm,
            ln1_scale,
            ln1_bias,
            ln2_scale,
            ln2_bias,
            epsilon,
            qkv_bias_tensor,
            out_linear_bias,
            cache_kv,
            attn_mask,
            self.dropout_prob,
            self.attn_dropout_prob,
            ln2_epsilon,
        )

        if self.has_cache_kv:
            return final_out[0], final_out[1]

        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True
        )
        return final_out, x.grad

    def test_fused_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=self.rtol, atol=self.atol
        )
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=self.rtol, atol=self.atol
        )


if __name__ == "__main__":
    unittest.main()
