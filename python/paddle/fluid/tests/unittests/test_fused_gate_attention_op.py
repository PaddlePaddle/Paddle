# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.incubate.nn.functional as F
from paddle import tensor
import unittest
from op_test import OpTest
from paddle.fluid.framework import default_main_program

default_main_program().random_seed = 42


class TestFusedGateAttentionOp(OpTest):
    def setUp(self):
        self.config()
        self.generate_input_data()
        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_gate_attention"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = True
        self.query_w = paddle.create_parameter(
            [self.qkv_dim, self.num_heads, self.c],
            paddle.get_default_dtype(),
            default_initializer=nn.initializer.XavierUniform())
        self.key_w = paddle.create_parameter(
            [self.qkv_dim, self.num_heads, self.c],
            paddle.get_default_dtype(),
            default_initializer=nn.initializer.XavierUniform())
        self.value_w = paddle.create_parameter(
            [self.qkv_dim, self.num_heads, self.c],
            paddle.get_default_dtype(),
            default_initializer=nn.initializer.XavierUniform())
        if self.bias_attr:
            self.nonbatched_bias = paddle.create_parameter(
                [
                    self.batch_size, self.num_heads, self.res_length,
                    self.res_length
                ],
                paddle.get_default_dtype(),
                default_initializer=nn.initializer.XavierUniform())
        if self.is_gating:
            self.gating_w = paddle.create_parameter(
                [self.qkv_dim, self.num_heads, self.c],
                paddle.get_default_dtype(),
                default_initializer=nn.initializer.XavierUniform())
            self.gating_b = paddle.create_parameter(
                [self.num_heads, self.c],
                paddle.get_default_dtype(),
                default_initializer=nn.initializer.XavierUniform())

        self.output_w = paddle.create_parameter(
            [self.num_heads, self.c, self.out_dim],
            paddle.get_default_dtype(),
            default_initializer=nn.initializer.XavierUniform())
        self.output_b = paddle.create_parameter(
            [self.out_dim],
            paddle.get_default_dtype(),
            default_initializer=nn.initializer.XavierUniform())

    def config(self):
        self.x_type = np.float32
        self.is_gating = True
        self.has_attn_mask = True
        self.training = True

        self.batch_size = 1
        self.msa_length = 3
        self.res_length = 2
        self.qkv_dim = 4

        self.num_heads = 8
        self.c = 4
        self.out_dim = self.qkv_dim
        self.bias_attr = True

    def generate_input_data(self):
        self.query = np.random.rand(self.batch_size, self.msa_length,
                                    self.res_length,
                                    self.qkv_dim).astype(self.x_type)

        self.attn_mask = np.ones(
            (self.batch_size, self.msa_length, self.res_length),
            dtype=self.x_type)
        self.tensor_query = paddle.to_tensor(self.query, stop_gradient=False)
        attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        bias = 1e3 * (attn_mask - 1.)
        self.bias = paddle.unsqueeze(bias, axis=[2, 3])

        self.dout = np.random.random(
            (self.batch_size, self.msa_length, self.res_length,
             self.c)).astype(self.x_type)

    def GetBaselineOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        c = self.c**(-0.5)
        # print(tensor_query.shape)
        tensor_query = self.tensor_query
        q = paddle.einsum('nbqa,ahc->nbqhc', tensor_query, self.query_w) * c
        k = paddle.einsum('nbqa,ahc->nbqhc', tensor_query, self.key_w)
        v = paddle.einsum('nbqa,ahc->nbqhc', tensor_query, self.value_w)
        # print(q.shape)
        # print(k.shape)
        # print(bias.shape)
        logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + self.bias

        if self.bias_attr:
            logits += paddle.unsqueeze(self.nonbatched_bias, axis=1)

        weights = nn.functional.softmax(logits)
        weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

        if self.is_gating:
            gate_values = paddle.einsum('nbqc,chv->nbqhv', tensor_query,
                                        self.gating_w) + self.gating_b
            gate_values = nn.functional.sigmoid(gate_values)
            weighted_avg *= gate_values

        final_out = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                                  self.output_w) + self.output_b
        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        return final_out, tensor_query.grad

    def GetFusedGateAttentionOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = self.tensor_query
        qkv_weight = paddle.stack(
            [self.query_w, self.key_w, self.value_w], axis=0)
        qkv_weight = paddle.transpose(qkv_weight, perm=[0, 2, 3, 1])

        if self.bias_attr:
            nonbatched_bias = self.nonbatched_bias
        else:
            nonbatched_bias = None

        if self.is_gating:
            gating_w = self.gating_w
            gating_b = self.gating_b
        else:
            gating_w = None
            gating_b = None

        final_out = F.fused_gate_attention(
            x=x,
            qkv_weight=qkv_weight,
            linear_weight=self.output_w,
            gate_weight=gating_w,
            linear_bias=self.output_b,
            gate_bias=gating_b,
            nonbatched_bias=nonbatched_bias,
            attn_mask=self.bias,
            is_gating=self.is_gating)

        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        return final_out, x.grad

    def test_fused_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedGateAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-3)


class TestFusedGateAttentionOpBiasIsNone(TestFusedGateAttentionOp):
    def config(self):
        super().config()
        self.bias_attr = False


class TestFusedGateAttentionOpPreLn(TestFusedGateAttentionOp):
    def config(self):
        super().config()
        self.is_gating = False


class TestFusedGateAttentionOpFp16(TestFusedGateAttentionOp):
    def config(self):
        super().config()
        self.x_type = np.float16

    def test_fused_gate_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedGateAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
