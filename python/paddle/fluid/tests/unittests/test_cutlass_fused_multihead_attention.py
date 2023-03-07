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
import math
import os
# from turtle import forward
import unittest
from paddle.nn.functional import loss
import paddle

from paddle.tensor.math import scale
import paddle.fluid.core as core

import numpy as np
from op_test import OpTest

# Ensure we use float type to accumulate
os.environ["FLAGS_gemm_use_half_precision_compute_type"] = "0"

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.fluid.framework import default_main_program
from paddle.incubate.nn.functional import cutlass_fused_multi_head_attention, cutlass_fused_multi_head_attention_grad 

default_main_program().random_seed = 42

class TestCutlassFMHAOp(OpTest):
    def setUp(self):
        self.config()
        self.rtol = 1e-3
        self.atol = 1e-4
        self.paddle_dtype = paddle.float32
        if self.x_type == np.float16:
            self.paddle_dtype = paddle.float16

        paddle.seed(2021)

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "cutlass_fused_multihead_attention"
        # Since it's only used in inference.
        self.__class__.no_need_check_grad = False

        if (self.paddle_dtype == paddle.float32):
            self.query_static = np.random.uniform(
                    low=-0.1,
                    high=0.1,
                    size=[
                        self.batch,
                        self.query_seq_len,
                        self.num_head,
                        self.head_size,
                    ],
                ).astype('float32')
            self.query = paddle.to_tensor(
                self.query_static,
                dtype=self.paddle_dtype,
            )

            self.key_static = np.random.uniform(
                    low=-0.1,
                    high=0.1,
                    size=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ],
                ).astype('float32')
            self.key = paddle.to_tensor(
                self.key_static,
                dtype=self.paddle_dtype,
            )

            self.value_static = np.random.uniform(
                    low=-0.1,
                    high=0.1,
                    size=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ],
                ).astype('float32')
            self.value = paddle.to_tensor(
                self.value_static,
                dtype=self.paddle_dtype,
            )

            self.mask_static = np.random.uniform(
                    low=-0.1, 
                    high=0.1, 
                    size=self.mask_shape,
                    ).astype('float32')
            self.mask = paddle.to_tensor(
                self.mask_static,
                dtype=self.paddle_dtype,
            )

            self.predect_static = np.random.uniform(
                    low=-0.1,
                    high=0.1,
                    size=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ],
                ).astype('float32')
            self.predect = paddle.to_tensor(
                self.predect_static,
                dtype=self.paddle_dtype,)

        elif (self.paddle_dtype == paddle.float16):
            self.query_static = np.random.uniform(
                    low=-0.1,
                    high=0.1,
                    size=[
                        self.batch,
                        self.query_seq_len,
                        self.num_head,
                        self.head_size,
                    ],
                ).astype('float16')
            self.query = paddle.to_tensor(
                self.query_static,
                dtype=self.paddle_dtype,
            )

            self.key_static = np.random.uniform(
                    low=-0.1,
                    high=0.1,
                    size=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ],
                ).astype('float16')
            self.key = paddle.to_tensor(
                self.key_static,
                dtype=self.paddle_dtype,
            )

            self.value_static = np.random.uniform(
                    low=-0.1,
                    high=0.1,
                    size=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ],
                ).astype('float16')
            self.value = paddle.to_tensor(
                self.value_static,
                dtype=self.paddle_dtype,
            )

            self.mask_static = np.random.uniform(
                    low=-0.1, 
                    high=0.1, 
                    size=self.mask_shape,
                    ).astype('float32')
            self.mask = paddle.to_tensor(
                self.mask_static,
                dtype=self.paddle_dtype,
            )

            self.predect_static = np.random.uniform(
                    low=-0.1,
                    high=0.1,
                    size=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ],
                ).astype('float16')
            self.predect = paddle.to_tensor(
                self.predect_static,
                dtype=self.paddle_dtype,)
        else:
            assert("paddle_dtype type is not supported")

        self.query.stop_gradient = False
        self.key.stop_gradient = False
        self.value.stop_gradient = False


    def config(self):
        self.x_type = np.float16
        self.batch = 1
        self.num_head = 8
        self.query_seq_len = 128
        self.kv_seq_len = 128
        self.head_size = 32
        self.scale = float(1.0 / math.sqrt(self.head_size))
        self.causal = True
        self.mask_shape = [1, 1, 1, self.kv_seq_len]
        self.dropout_p = 0.0

    def GetBaselineOut(self):
        paddle.disable_static()

        self.add_mask = isinstance(self.mask, paddle.Tensor)
        self.use_dropout = self.dropout_p!=0.0

        assert(isinstance(self.query,paddle.Tensor))
        query = paddle.transpose(self.query, [0, 2, 1, 3])
        key = paddle.transpose(self.key, [0, 2, 1, 3])
        value = paddle.transpose(self.value, [0, 2, 1, 3])

        qk_res = paddle.matmul(query, key, transpose_y=True)
        attention = qk_res * self.scale
        if self.add_mask:
            attention = attention + self.mask

        softmax_result = paddle.nn.functional.softmax(attention, -1)
        result = paddle.matmul(softmax_result, value)
        if (self.use_dropout):
            result = F.dropout(softmax_result, self.dropout_p)
        result = paddle.transpose(result, [0, 2, 1, 3])

        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        loss = paddle.mean(result - self.predect)
        loss.stop_gradient = False
        loss.backward()

        g_q_base = query.grad
        g_k_base = key.grad
        g_v_base = value.grad

        return result, g_q_base, g_k_base, g_v_base

    def GetFusedMultiheadAttentionOut(self):

        paddle.enable_static()
        main_prog = paddle.static.Program()

        print("GetFusedMultiheadAttentionOut program constructing:")
        with paddle.static.program_guard(main_prog):
            query = paddle.static.data(name='query', dtype='float32', shape=[
                        self.batch,
                        self.query_seq_len,
                        self.num_head,
                        self.head_size,
                    ])
            key = paddle.static.data(name='key', dtype='float32', shape=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ])
            value = paddle.static.data(name='value', dtype='float32', shape=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ])
            predect = paddle.static.data(name='predect', dtype='float32', shape=[
                        self.batch,
                        self.kv_seq_len,
                        self.num_head,
                        self.head_size,
                    ])
            mask = paddle.static.data(name='mask', dtype='float32', shape=[
                1, 1, 1, self.kv_seq_len])

            scale = paddle.static.data(name='scale', dtype='float32', shape=[1])
            causal = paddle.static.data(name='causal', dtype='bool', shape=[1])
            dropout_p = paddle.static.data(name='dropout_p', dtype='float32', shape=[1])

            query.stop_gradient = False
            key.stop_gradient = False
            value.stop_gradient = False

            scale.stop_gradient = True
            causal.stop_gradient = True
            dropout_p.stop_gradient = True

            out, seed_and_offset= cutlass_fused_multi_head_attention(
                query,
                key,
                value,
                mask,
                self.scale,
                self.causal,
                self.dropout_p
            )        

            # import pdb; pdb.set_trace()
            loss = paddle.mean(out-predect)
            paddle.static.append_backward(loss)
                
            fetch_list = [out.name,
                            "query_grad",
                            "key_grad", 
                            "value_grad"]
            # fetch_list = [fused_out]
                        
            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            fused_out, q_g, k_g, v_g = exe.run(main_prog, 
                feed = {"query" : self.query_static,
                        "key" : self.key_static,
                        "value" : self.value_static,
                        "predect" : self.predect_static,
                        "mask" : self.mask_static,
                        "scale" : self.scale,
                        "causal" : self.causal,
                        "dropout_p" : self.dropout_p},
                fetch_list=fetch_list)
        
        print("GetFusedMultiheadAttentionOut program has been constructed")
        paddle.disable_static()

        return fused_out, q_g, k_g, v_g

    def test_baseline_op(self):
        self.setUp()
        fused_out_ref, g_q_ref, g_k_ref, g_v_ref = self.GetBaselineOut()
        fused_out, g_q, g_k, g_v = self.GetFusedMultiheadAttentionOut()
        np.testing.assert_allclose(
            fused_out_ref, fused_out, rtol=self.rtol, atol=self.atol
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

class TestCutlassFMHAMaskOpDatatype(TestCutlassFMHAOp):
    def config(self):
        super().config()
        self.x_type = np.float32
        if (self.x_type == np.float16):
            self.batch = 8
            self.num_head = 8
            self.query_seq_len = 1024
            self.kv_seq_len = 1024
            self.head_size = 32
            self.scale = float(1.0 / math.sqrt(self.head_size))
            self.causal = False
            self.mask_shape = [1, 1, 1, self.kv_seq_len]
            self.dropout_p = 0.0
        elif (self.x_type == np.float32):
            self.batch = 1
            self.num_head = 8
            self.query_seq_len = 32
            self.kv_seq_len = 32
            self.head_size = 32
            self.scale = float(1.0 / math.sqrt(self.head_size))
            self.causal = False
            self.mask_shape = [1, 1, 1, self.kv_seq_len]
            self.dropout_p = 0.0
        else :
            assert(f"this type ({self.x_type}) is not supported")

if __name__ == "__main__":
    unittest.main()
