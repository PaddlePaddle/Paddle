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

from asyncio.windows_events import NULL
import math
import os
from turtle import forward
import unittest
from Paddle.python.paddle.tensor.math import scale

import numpy as np
from op_test import OpTest

# Ensure we use float type to accumulate
os.environ["FLAGS_gemm_use_half_precision_compute_type"] = "0"

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.fluid.framework import default_main_program
from paddle.incubate.nn.functional import cutlass_fused_multi_head_attention

default_main_program().random_seed = 42

class PaddleFMHAOp(nn.Layer):
    def __init__(
        self,
        scale,
        add_mask,
        mask,
        use_dropout,
        dropout_p
    ) -> None:
        super().__init__()
        self.scale = scale
        self.add_mask = add_mask
        self.mask = mask
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
    
    def forward(self, q, k, v):
        paddle.disable_static()
        # Transpose to (batch, num_head, seq_len, head_size)
        query = paddle.transpose(q, [0, 2, 1, 3])
        key = paddle.transpose(k, [0, 2, 1, 3])
        value = paddle.transpose(v, [0, 2, 1, 3])

        qk_res = paddle.matmul(query, key, transpose_y=True)
        attention = qk_res * self.scale
        if self.add_mask:
            attention = attention + self.mask

        softmax_result = paddle.nn.functional.softmax(attention, -1)
        result = paddle.matmul(softmax_result, value)
        if (self.use_dropout):
            result = F.dropout(softmax_result, self.dropout_p)
        result = paddle.transpose(result, [0, 2, 1, 3])
        return result
        
        

class TestCutlassFMHAOp(OpTest):
    def setUp(self):
        self.config()
        self.use_dropout = False
        self.rtol = 1e-3
        self.atol = 1e-4

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "cutlass_fused_multihead_attention"
        # Since it's only used in inference.
        self.__class__.no_need_check_grad = True

        self.query = paddle.to_tensor(
            np.random.uniform(
                low=-0.01,
                high=0.01,
                size=[
                    self.batch,
                    self.query_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            ),
            dtype=self.datatype,
        )
        self.key = paddle.to_tensor(
            np.random.uniform(
                low=-0.01,
                high=0.01,
                size=[
                    self.batch,
                    self.kv_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            ),
            dtype=self.datatype,
        )
        self.value = paddle.to_tensor(
            np.random.uniform(
                low=-0.01,
                high=0.01,
                size=[
                    self.batch,
                    self.kv_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            ),
            dtype=self.datatype,
        )
        self.mask = paddle.to_tensor(
            np.random.uniform(low=-0.01, high=0.01, size=self.mask_shape),
            dtype=self.datatype,
        )

        self.query.stop_gradient = False
        self.key.stop_gradient = False
        self.value.stop_gradient = False

        self.out_delt = paddle.to_tensor(
            np.random.uniform(
                low=-0.01,
                high=0.01,
                size=[
                    self.batch,
                    self.query_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            ),
            dtype=self.datatype,
        )

        self.out_grad = paddle.to_tensor(
            np.random.uniform(
                low=-0.01,
                high=0.01,
                size=[
                    self.batch,
                    self.query_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            ),
            dtype=self.datatype,
        )

    def config(self):
        self.x_type = np.float16
        self.datatype = paddle.float32
        self.batch = 1
        self.num_head = 8
        self.query_seq_len = 1024
        self.kv_seq_len = 1024
        self.head_size = 32
        self.scale = float(1.0 / math.sqrt(self.head_size))
        self.add_mask = True
        self.dropout_p = 0.5
        self.mask_shape = [
            self.batch,
            self.num_head,
            self.query_seq_len,
            self.kv_seq_len,
        ]
        self.out_delt = 1e-5

    def GetBaselineOut(self):
        paddle.disable_static()
        baselinefmha = PaddleFMHAOp(
            self.scale,
            self.add_mask,
            self.mask,
            self.use_dropout,
            self.dropout_p
        )

        query = self.query
        key = self.key
        value = self.value

        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False

        result = baselinefmha.forward(
            query,
            key,
            value)    
        result.stop_gradient = False

        paddle.autograd.backward(result, self.out_grad, create_graph=True)

        g_q_base = query.grad
        g_k_base = key.grad
        g_v_base = value.grad

        return result, g_q_base, g_k_base, g_v_base

    def GetFusedMultiheadAttentionOut(self):
        paddle.disable_static()

        if (self.use_dropout): 
            dropout_p = self.dropout_p     
        else:
            dropout_p = NULL
      
        fused_out, seed_and_offset = cutlass_fused_multi_head_attention(
            self.query,
            self.key,
            self.value,
            self.mask if self.add_mask else None,
            self.scale,
            dropout_p,
        )

        g_q, g_k, g_v = cutlass_fused_multi_head_attention_grad(
            self.query,
            self.key,
            self.value,
            seed_and_offset,
            fused_out,
            self.out_delt,
            self.out_grad,
            self.add_mask,
            self.scale,
            dropout_p,
        )

        return fused_out, g_q, g_k, g_v

    def test_fused_multihead_attention_op(self):
        fused_out, g_q, g_k, g_v = self.GetFusedMultiheadAttentionOut()
        final_out_ref, g_q_ref, g_k_ref, g_v_ref = self.GetBaselineOut()

        np.testing.assert_allclose(
            final_out_ref, fused_out, rtol=self.rtol, atol=self.atol
        )
        np.testing.assert_allclose(
            g_q, g_q_ref, rtol=self.rtol, atol=self.atol
        )
        np.testing.assert_allclose(
            g_k, g_k_ref, rtol=self.rtol, atol=self.atol
        )
        np.testing.assert_allclose(
            g_v, g_v_ref, rtol=self.rtol, atol=self.atol
        )


class TestCutlassFMHAOpFp16(TestCutlassFMHAOp):
    def config(self):
        super().config()
        self.x_type = np.float16
        self.batch = 4
        self.num_head = 8
        self.query_seq_len = 256
        self.kv_seq_len = 256
        self.head_size = 64
        self.scale = float(1.0 / math.sqrt(self.head_size))
        self.add_mask = True
        self.dropout_p = 0.5
        self.mask_shape = [
            self.batch,
            self.num_head,
            self.query_seq_len,
            self.kv_seq_len,
        ]
        self.out_delt = 1e-5


class TestCutlassFMHAMaskOpFp16(TestCutlassFMHAOp):
    def config(self):
        super().config()
        self.x_type = np.float16
        self.batch = 1
        self.num_head = 8
        self.query_seq_len = 900
        self.kv_seq_len = 6000
        self.head_size = 32
        self.scale = float(1.0 / math.sqrt(self.head_size))
        self.add_mask = True
        self.dropout_p = 0.5
        self.mask_shape = [1, 1, 1, self.kv_seq_len]
        self.out_delt = 1e-5


class TestCutlassFMHAOpFp32(TestCutlassFMHAOp):
    def config(self):
        super().config()
        self.x_type = np.float32
        self.batch = 8
        self.num_head = 8
        self.query_seq_len = 128
        self.kv_seq_len = 128
        self.head_size = 32
        self.scale = float(1.0 / math.sqrt(self.head_size))
        self.add_mask = True
        self.dropout_p = 0.5
        self.mask_shape = [
            self.batch,
            self.num_head,
            self.query_seq_len,
            self.kv_seq_len,
        ]
        self.out_delt = 1e-5


if __name__ == "__main__":
    unittest.main()
