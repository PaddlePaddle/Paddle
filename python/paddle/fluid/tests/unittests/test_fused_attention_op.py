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

import copy
import collections
import numpy as np

import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn.functional as F
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout
from paddle.fluid.data_feeder import convert_dtype
#from paddle.nn.layer import transformer
from paddle import tensor
from paddle.fluid import layers

import unittest

place = paddle.CUDAPlace(0)


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.

    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.

    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedAttentionOp(unittest.TestCase):
    def setUp(self):
        self.config()
        self.generate_input_data()
        paddle.set_default_dtype(self.x_type)
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.k_proj = Linear(
            self.kdim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.v_proj = Linear(
            self.vdim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        paddle.set_default_dtype(np.float32)
        self.norm1 = LayerNorm(self.embed_dim)
        self.norm2 = LayerNorm(self.embed_dim)
        paddle.set_default_dtype(self.x_type)
        self.dropout = Dropout(self.dropout_prob, mode="upscale_in_train")

    def config(self):
        self.x_type = np.float32
        self.attn_mask_type = np.float64
        self.pre_layer_norm = True
        self.training = True

        # self.batch_size = 1
        # self.query_length = 2
        # self.head_dim = 2
        # self.num_heads = 1
        self.batch_size = 8
        self.query_length = 128
        self.head_dim = 64
        self.num_heads = 16
        self.embed_dim = self.head_dim * self.num_heads

        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        #self.self_attention = True

        #self.need_weight = False
        #self.param_attr = None
        self.weight_attr = None
        self.bias_attr = None

        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = self.query_length, self.query_length
        # if self.self_attention:
        #     self.kdim, self.vdim = self.embed_dim, self.embed_dim
        #     self.key_length, self.value_length = self.query_length, self.query_length
        # else:
        #     ## todo
        #     self.kdim, self.vdim = [np.random.randint(5, 20) for _ in range(2)]
        #     self.key_length = np.random.randint(2, 10)
        #     self.value_length = key_length

    def generate_input_data(self):
        self.query = np.random.rand(self.batch_size, self.query_length,
                                    self.embed_dim).astype(self.x_type)

        # [bs, 16, 128, 128]
        self.attn_mask = np.ones(
            (self.batch_size, self.num_heads, self.query_length,
             self.key_length),
            dtype=self.attn_mask_type)

        if self.attn_mask_type == np.int64:
            self.attn_mask = np.tril(self.attn_mask)
        elif self.attn_mask_type == np.float64:
            self.attn_mask = (np.tril(self.attn_mask) - 1.0) * 1e9
        else:
            raise ValueError("'attn_mask_type' should be 'int64' or 'float64'.")

        self.key, self.value = self.query, self.query
        # if self.self_attention:
        #     self.key, self.value = self.query, self.query
        # else:
        #     self.key = np.random.rand(batch_size, key_length, kdim).astype(self.x_type)
        #     self.value = np.random.rand(batch_size, value_length, vdim).astype(self.x_type)

        self.dout = np.random.random((self.batch_size, self.query_length,
                                      self.embed_dim)).astype(self.x_type)

    def GetBaselineOut(self):
        tensor_query = paddle.to_tensor(self.query, stop_gradient=False)
        attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        #residual = paddle.to_tensor(self.query)
        residual = tensor_query

        for i in range(1):
            ln1_out = tensor_query
            # pre_layernorm
            if self.pre_layer_norm:
                ln1_out = self.norm1(tensor_query)

            # print("ln1_out.dtype")
            # print(ln1_out.dtype)

            # get q, k, v
            q = self.q_proj(ln1_out)
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q_out = tensor.transpose(x=q, perm=[0, 2, 1, 3])
            k = self.k_proj(ln1_out)
            v = self.v_proj(ln1_out)
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k_out = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v_out = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            # q_out * k^t
            qk_out = layers.matmul(
                x=q_out, y=k_out, transpose_y=True, alpha=self.head_dim**-0.5)

            if attn_mask is not None:
                # Support bool or int mask
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
                    mode="upscale_in_train")
                qktv_out = tensor.matmul(dropout_out, v_out)
            else:
                qktv_out = tensor.matmul(softmax_out, v_out)

            # combine heads
            fmha_out = tensor.transpose(qktv_out, perm=[0, 2, 1, 3])

            out_linear_in = tensor.reshape(
                x=fmha_out,
                shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]])
            # project to output
            out = self.out_proj(out_linear_in)

            # #
            residual_out = residual + self.dropout(out)
            if not self.pre_layer_norm:
                final_out = self.norm1(residual_out)
            if self.pre_layer_norm:
                final_out = self.norm2(residual_out)

            paddle.autograd.backward(
                [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
            # return final_out, tensor_query.grad

        #paddle.autograd.backward([out], [paddle.to_tensor(self.dout)])
        return ln1_out, softmax_out, attn_mask_out, out, final_out, final_out.grad, residual_out.grad, out.grad, fmha_out.grad, qktv_out.grad, softmax_out.grad, qk_out.grad, ln1_out.grad, tensor_query.grad
        #return ln1_out, softmax_out, attn_mask_out, out, final_out

    def GetFusedAttentionOut(self):
        #with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        q_proj_weight = paddle.to_tensor(
            self.q_proj.weight, stop_gradient=False)
        q_proj_bias = paddle.to_tensor(self.q_proj.bias, stop_gradient=False)
        k_proj_weight = paddle.to_tensor(
            self.k_proj.weight, stop_gradient=False)
        k_proj_bias = paddle.to_tensor(self.k_proj.bias, stop_gradient=False)
        v_proj_weight = paddle.to_tensor(
            self.v_proj.weight, stop_gradient=False)
        v_proj_bias = paddle.to_tensor(self.v_proj.bias, stop_gradient=False)
        out_linear_weight = paddle.to_tensor(
            self.out_proj.weight, stop_gradient=False)
        out_linear_bias = paddle.to_tensor(
            self.out_proj.bias, stop_gradient=False)

        ln1_scale = paddle.to_tensor(self.norm1.weight, stop_gradient=False)
        ln1_bias = paddle.to_tensor(self.norm1.bias, stop_gradient=False)
        ln2_scale = paddle.to_tensor(self.norm2.weight, stop_gradient=False)
        ln2_bias = paddle.to_tensor(self.norm2.bias, stop_gradient=False)

        print(q_proj_weight.shape)
        print(q_proj_bias.shape)

        q_proj_weight = q_proj_weight.numpy().transpose((1, 0))
        k_proj_weight = k_proj_weight.numpy().transpose((1, 0))
        v_proj_weight = v_proj_weight.numpy().transpose((1, 0))
        qkv_weight = np.concatenate((q_proj_weight, k_proj_weight))
        qkv_weight = np.concatenate((qkv_weight, v_proj_weight))
        #print(qkv_weight.shape)
        #[3, num_heads, self.head_dim, embed_dim]
        qkv_weight = qkv_weight.reshape(
            (3, self.num_heads, self.head_dim, self.embed_dim))
        #print(qkv_weight.shape)

        qkv_bias = np.concatenate((q_proj_bias.numpy(), k_proj_bias.numpy()))
        qkv_bias = np.concatenate((qkv_bias, v_proj_bias.numpy()))
        #print(qkv_bias.shape)
        #[3, num_heads, self.head_dim]
        qkv_bias = qkv_bias.reshape((3, self.num_heads, self.head_dim))
        #print(qkv_bias.shape)

        x = paddle.to_tensor(self.query, stop_gradient=False)
        attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        qkv_weight_tensor = paddle.to_tensor(qkv_weight, stop_gradient=False)
        qkv_bias_tensor = paddle.to_tensor(qkv_bias, stop_gradient=False)
        epsilon = 1e-05
        ln2_epsilon = 1e-05

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, x.dtype)

        for i in range(1):
            ln_out, qkv_out, qkv_bias_out, transpose_out_2, qk_out, qktv_out, softmax_out, src_mask_out, fmha_out, out_linear_out, bias_dropout_residual_out, final_out = F.fused_multihead_attention(
                x, qkv_weight_tensor, out_linear_weight, self.pre_layer_norm,
                ln1_scale, ln1_bias, ln2_scale, ln2_bias, epsilon,
                qkv_bias_tensor, out_linear_bias, attn_mask, self.dropout_prob,
                self.attn_dropout_prob, ln2_epsilon)
            # ln_out, qkv_out, qkv_bias_out, transpose_out_2, qk_out, qktv_out, softmax_out, src_mask_out, fmha_out, final_out = F.fused_multihead_attention(x, qkv_weight_tensor, out_linear_weight, 
            #                                 self.pre_layer_norm, ln1_scale, 
            #                                 ln1_bias, epsilon, qkv_bias_tensor, out_linear_bias, 
            #                                 attn_mask, self.dropout_prob)
            paddle.autograd.backward(
                [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)

        #return out, x.grad
        #return ln_out, softmax_out, src_mask_out, out_linear_out, bias_dropout_residual_out, final_out, bias_dropout_residual_out.grad, out_linear_out.grad, fmha_out.grad

        return ln_out, softmax_out, src_mask_out, out_linear_out, final_out, final_out.grad, bias_dropout_residual_out.grad, out_linear_out.grad, fmha_out.grad, qktv_out.grad, softmax_out.grad, qk_out.grad, ln_out.grad, x.grad
        #return ln_out, softmax_out, src_mask_out, out_linear_out, final_out

    def test_fused_attention_op(self):
        print(
            "self.batch_size, self.query_length, self.embed_dim, self.num_heads, self.head_dim"
        )
        print(self.batch_size, self.query_length, self.embed_dim,
              self.num_heads, self.head_dim)
        #base_out, base_grad = self.GetBaselineOut()
        ln_out_ref, softmax_out_ref, src_mask_out_ref, out_linear_out_ref, final_out_ref, final_out_grad_ref, residual_grad_ref, out_linear_grad_ref, fmha_grad_ref, qktv_grad_ref, softmax_grad_ref, qk_grad_ref, ln_grad_ref, x_grad_ref = self.GetBaselineOut(
        )

        # #fused_out, fused_grad = self.GetFusedAttentionOut()
        ln_out, softmax_out, src_mask_out, out_linear_out, final_out, final_out_grad, bias_dropout_residual_out_grad, out_linear_grad, fmha_grad, qktv_grad, softmax_grad, qk_grad, ln_grad, x_grad = self.GetFusedAttentionOut(
        )

        np.testing.assert_allclose(
            ln_out_ref, ln_out.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            softmax_out_ref, softmax_out.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            src_mask_out_ref, src_mask_out.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            out_linear_out_ref, out_linear_out.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(
            final_out_grad_ref, final_out_grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            residual_grad_ref,
            bias_dropout_residual_out_grad.numpy(),
            rtol=1e-5,
            atol=1e-5)
        np.testing.assert_allclose(
            out_linear_grad_ref, out_linear_grad.numpy(), rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(
            fmha_grad_ref, fmha_grad.numpy(), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(
            qktv_grad_ref, qktv_grad.numpy(), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(
            softmax_grad_ref, softmax_grad.numpy(), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(
            qk_grad_ref, qk_grad.numpy(), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(
            ln_grad_ref, ln_grad.numpy(), rtol=1e-5, atol=1e-4)

        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-4)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedAttentionOpFp16(TestFusedAttentionOp):
    def config(self):
        self.x_type = np.float16
        self.attn_mask_type = np.float64
        self.pre_layer_norm = True

        self.batch_size = 8
        self.query_length = 128
        self.head_dim = 64
        self.num_heads = 16
        self.embed_dim = self.head_dim * self.num_heads

        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0

        self.weight_attr = None
        self.bias_attr = None

        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = self.query_length, self.query_length

    def test_fused_attention_op(self):
        print(
            "self.batch_size, self.query_length, self.embed_dim, self.num_heads, self.head_dim"
        )
        print(self.batch_size, self.query_length, self.embed_dim,
              self.num_heads, self.head_dim)
        #base_out, base_grad = self.GetBaselineOut()
        ln_out_ref, softmax_out_ref, src_mask_out_ref, out_linear_out_ref, final_out_ref, final_out_grad_ref, residual_grad_ref, out_linear_grad_ref, fmha_grad_ref, qktv_grad_ref, softmax_grad_ref, qk_grad_ref, ln_grad_ref, x_grad_ref = self.GetBaselineOut(
        )
        #fused_out, fused_grad = self.GetFusedAttentionOut()
        ln_out, softmax_out, src_mask_out, out_linear_out, final_out, final_out_grad, bias_dropout_residual_out_grad, out_linear_grad, fmha_grad, qktv_grad, softmax_grad, qk_grad, ln_grad, x_grad = self.GetFusedAttentionOut(
        )

        np.testing.assert_allclose(
            ln_out_ref, ln_out.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            softmax_out_ref, softmax_out.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            src_mask_out_ref, src_mask_out.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            out_linear_out_ref, out_linear_out.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-1)

        np.testing.assert_allclose(
            final_out_grad_ref, final_out_grad.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            residual_grad_ref,
            bias_dropout_residual_out_grad.numpy(),
            rtol=1e-5,
            atol=1e-1)
        np.testing.assert_allclose(
            out_linear_grad_ref, out_linear_grad.numpy(), rtol=1e-5, atol=1e-1)

        np.testing.assert_allclose(
            fmha_grad_ref, fmha_grad.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            qktv_grad_ref, qktv_grad.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            softmax_grad_ref, softmax_grad.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            qk_grad_ref, qk_grad.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            ln_grad_ref, ln_grad.numpy(), rtol=1e-5, atol=1e-1)

        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
