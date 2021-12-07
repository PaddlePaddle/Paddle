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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.fluid.core as core
import paddle.nn.functional as F
import paddle.incubate.nn.functional as incubate_f
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout
# from paddle.fluid.data_feeder import convert_dtype
from paddle import tensor
from paddle.fluid import layers

import unittest

# place = paddle.CUDAPlace(0)

@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedAttentionCuDNNFMHAOp(unittest.TestCase):
    def setUp(self):
        self.setXType()
        self.setInputSize()
        self.setDropoutRate()
        self.setPreLn()
        self.setBiasAttr()
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
        paddle.set_default_dtype(self.x_type)
        self.dropout = Dropout(self.dropout_prob, mode="upscale_in_train")

    def setBiasAttr(self):
        self.bias_attr = None

    def setPreLn(self):
        self.pre_layer_norm = False

    def setXType(self):
        self.x_type = np.float32
    
    def setInputSize(self):
        self.batch_size = 8
        self.query_length = 128
        self.head_dim = 64
        self.num_heads = 16

    def setDropoutRate(self):
        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0

    def config(self):
        self.training = True
        self.embed_dim = self.head_dim * self.num_heads

        self.weight_attr = None
        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = self.query_length, self.query_length

    def generate_input_data(self):
        self.query = np.random.rand(self.batch_size, self.query_length,
                                    self.embed_dim).astype(self.x_type)
        self.key, self.value = self.query, self.query

        self.seq_len_vec = np.full((self.batch_size, ), self.query_length, dtype=np.int32)
        self.attn_low_windows = np.zeros((self.query_length, ), dtype=np.int32)
        self.attn_high_windows = np.full((self.query_length, ), self.query_length, dtype=np.int32)

        self.dout = np.random.random((self.batch_size, self.query_length,
                                      self.embed_dim)).astype(self.x_type)

    def GetBaselineOut(self):
        tensor_query = paddle.to_tensor(self.query, stop_gradient=False)
        # attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        #seq_len_vec = paddle.to_tensor(self.seq_len_vec, stop_gradient=True) # device
        #residual = paddle.to_tensor(self.query)
        residual = tensor_query
        dout_tensor = paddle.to_tensor(self.dout)

        for i in range(1):
            ln1_out = tensor_query
            if self.pre_layer_norm:
                ln1_out = self.norm1(tensor_query)

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

            residual_out = residual + self.dropout(out)
            if not self.pre_layer_norm:
                final_out = self.norm1(residual_out)
            else:
                final_out = residual_out

            paddle.autograd.backward(
                [final_out], [dout_tensor], retain_graph=True)
        #return ln1_out, out, final_out, final_out.grad, out.grad, ln1_out.grad, tensor_query.grad
        #return out, final_out, out.grad, tensor_query.grad
        return final_out, tensor_query.grad

    def GetFusedAttentionCuDNNFMHAOut(self):
        # q_proj_weight = paddle.to_tensor(
        #     self.q_proj.weight, stop_gradient=False)
        #q_proj_bias = paddle.to_tensor(self.q_proj.bias, stop_gradient=False)
        # k_proj_weight = paddle.to_tensor(
        #     self.k_proj.weight, stop_gradient=False)
        #k_proj_bias = paddle.to_tensor(self.k_proj.bias, stop_gradient=False)
        # v_proj_weight = paddle.to_tensor(
        #     self.v_proj.weight, stop_gradient=False)
        #v_proj_bias = paddle.to_tensor(self.v_proj.bias, stop_gradient=False)
        # out_linear_weight = paddle.to_tensor(
        #     self.out_proj.weight, stop_gradient=False)
        # linear_bias = paddle.to_tensor(
        #     self.out_proj.bias, stop_gradient=False)

        ln_scale = paddle.to_tensor(self.norm1.weight, stop_gradient=False)
        ln_bias = paddle.to_tensor(self.norm1.bias, stop_gradient=False)

        # q_proj_weight = q_proj_weight.numpy().transpose((1, 0))
        # k_proj_weight = k_proj_weight.numpy().transpose((1, 0))
        # v_proj_weight = v_proj_weight.numpy().transpose((1, 0))
        # qkv_weight = np.concatenate((q_proj_weight, k_proj_weight))
        # qkv_weight = np.concatenate((qkv_weight, v_proj_weight))
        # qkv_weight = qkv_weight.reshape(
        #     (3, self.num_heads, self.head_dim, self.embed_dim))

        weight = np.concatenate((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.out_proj.weight))
        weight = weight.reshape((-1))

        if self.bias_attr is not False:
            bias = np.concatenate((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias, self.out_proj.bias))
            bias = bias.reshape((-1)) 
            has_bias = True
        else:
            bias = np.zeros((4*self.embed_dim,), dtype=self.x_type)
            has_bias = False

        cudnn_weight = np.concatenate((weight, bias))

        print("weight shape: ")
        print(weight.shape)

        print("cudnn_weight shape: ")
        print(cudnn_weight.shape)
    
        x = paddle.to_tensor(self.query, stop_gradient=False)
        seq_len_tensor = paddle.to_tensor(self.seq_len_vec, stop_gradient=True)
        #attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        weight_tensor = paddle.to_tensor(cudnn_weight, stop_gradient=False)
        dout_tensor = paddle.to_tensor(self.dout)
        epsilon = 1e-05
        # self.attn_low_windows
        attn_low_windows_tensor_host = paddle.to_tensor(self.attn_low_windows, place=paddle.CPUPlace())
        # self.attn_high_windows
        attn_high_windows_tensor_host = paddle.to_tensor(self.attn_high_windows, place=paddle.CPUPlace())
        # self.seq_len_vec
        seq_len_tensor_host = paddle.to_tensor(self.seq_len_vec, place=paddle.CPUPlace())

        # attn_low_windows_tensor_host = paddle.to_tensor(self.attn_low_windows)
        # # self.attn_high_windows
        # attn_high_windows_tensor_host = paddle.to_tensor(self.attn_high_windows)
        # # self.seq_len_vec
        # seq_len_tensor_host = paddle.to_tensor(self.seq_len_vec)

        # if attn_mask is not None:
        #     # Support bool or int mask
        #     attn_mask = _convert_attention_mask(attn_mask, x.dtype)

        for i in range(1):
            #ln_out, linear_out, final_out = F.fused_multihead_attention_cudnn_impl(
            final_out = incubate_f.fused_multihead_attention_cudnn_impl(
                x, weight_tensor, seq_len_tensor, self.num_heads, has_bias,
                self.pre_layer_norm, ln_scale, ln_bias, epsilon, 
                self.dropout_prob, self.attn_dropout_prob, 
                attn_low_windows_tensor_host, 
                attn_high_windows_tensor_host,
                seq_len_tensor_host, seq_len_tensor_host)
            paddle.autograd.backward(
                [final_out], [dout_tensor], retain_graph=True)

        #return ln_out, linear_out, final_out, final_out.grad, linear_out.grad, ln_out.grad, x.grad
        #return linear_out, final_out, linear_out.grad, x.grad
        return final_out, x.grad

    def test_fused_attention_cudnn_fmha_op(self):
        print(
            "self.batch_size, self.query_length, self.embed_dim, self.num_heads, self.head_dim = "
        )
        print(self.batch_size, self.query_length, self.embed_dim,
              self.num_heads, self.head_dim)
        #ln_out_ref, linear_out_ref, final_out_ref, final_grad_ref, linear_grad_ref, ln_grad_ref, x_grad_ref = self.GetBaselineOut()
        #ln_out, linear_out, final_out, final_grad, linear_grad, ln_grad, x_grad = self.GetFusedAttentionCuDNNFMHAOut()
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedAttentionCuDNNFMHAOut()
        
        #np.testing.assert_allclose(
        #    ln_out_ref, ln_out.numpy(), rtol=1e-5, atol=1e-5)
        # np.testing.assert_allclose(
        #    linear_out_ref, linear_out.numpy(), rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-3)

        #np.testing.assert_allclose(
        #    final_grad_ref, final_grad.numpy(), rtol=1e-5, atol=1e-4)
        # np.testing.assert_allclose(
        #    linear_grad_ref, linear_grad.numpy(), rtol=1e-5, atol=1e-3)
        #np.testing.assert_allclose(
        #    ln_grad_ref, ln_grad.numpy(), rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-3)


class TestFusedAttentionCuDNNFMHAOpPreLn(TestFusedAttentionCuDNNFMHAOp):
    def setPreLn(self):
        self.pre_layer_norm = True


class TestFusedAttentionCuDNNFMHAOpBiasIsNone(TestFusedAttentionCuDNNFMHAOp):
    def setBiasAttr(self):
        self.bias_attr = False


class TestFusedAttentionCuDNNFMHAOpFp16(TestFusedAttentionCuDNNFMHAOp):
    def setXType(self):
        self.x_type = np.float16

    def test_fused_attention_cudnn_fmha_op(self):
        print(
            "self.batch_size, self.query_length, self.embed_dim, self.num_heads, self.head_dim = "
        )
        print(self.batch_size, self.query_length, self.embed_dim,
              self.num_heads, self.head_dim)
        #ln_out_ref, linear_out_ref, final_out_ref, final_grad_ref, linear_grad_ref, ln_grad_ref, x_grad_ref = self.GetBaselineOut()
        #ln_out, linear_out, final_out, final_grad, linear_grad, ln_grad, x_grad = self.GetFusedAttentionCuDNNFMHAOut()
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedAttentionCuDNNFMHAOut()
        
        #np.testing.assert_allclose(
        #    ln_out_ref, ln_out.numpy(), rtol=1e-5, atol=1e-1)
        #np.testing.assert_allclose(
        #    linear_out_ref, linear_out.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-1)

        #np.testing.assert_allclose(
        #    final_grad_ref, final_grad.numpy(), rtol=1e-5, atol=1e-1)
        #np.testing.assert_allclose(
        #    linear_grad_ref, linear_grad.numpy(), rtol=1e-5, atol=1e-1)
        #np.testing.assert_allclose(
        #    ln_grad_ref, ln_grad.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-1)


class TestFusedAttentionCuDNNFMHAOpFp16BertLargeShape(TestFusedAttentionCuDNNFMHAOpFp16):
    def setInputSize(self):
        self.batch_size = 56
        self.query_length = 512
        self.head_dim = 64
        self.num_heads = 16


class TestFusedAttentionCuDNNFMHAOpFp16BiasIsNone(TestFusedAttentionCuDNNFMHAOpFp16):
    def setBiasAttr(self):
        self.bias_attr = False


if __name__ == "__main__":
    unittest.main()
