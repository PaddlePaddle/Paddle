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


import math
import os

# from turtle import forward
import unittest
from enum import Enum

import numpy as np
from op_test import OpTest

import paddle

# Ensure we use float type to accumulate
os.environ["FLAGS_gemm_use_half_precision_compute_type"] = "0"

import paddle.nn.functional as F
from paddle.fluid.framework import default_main_program
from paddle.incubate.nn.functional import cutlass_fused_multi_head_attention

default_main_program().random_seed = 42


class MaskType(Enum):
    NoMask = (0,)
    BlockDiagonalCausalMask = (1,)
    BlockDiagonalCausalWithOffsetPaddedKeysMask = (2,)
    BlockDiagonalMask = (3,)
    LowerTriangularMask = (4,)
    LowerTriangularMaskWithTensorBias = (5,)
    Random = 6


class TestCutlassFMHAOp(OpTest):
    def setUp(self):
        self.config()
        self.rtol = 1e-3
        self.atol = 1e-4
        if self.x_type == np.float16:
            self.paddle_dtype = paddle.float16

        paddle.seed(2021)

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "cutlass_fused_multihead_attention"
        # Since it's only used in inference.
        self.__class__.no_need_check_grad = False

        if self.paddle_dtype == paddle.float32:
            type = 'float32'
        elif self.paddle_dtype == paddle.float16:
            type = 'float16'
        else:
            assert "paddle_dtype type is not supported"

        self.GetQKV(type)
        self.GetBias(type)

    def config(self):
        self.paddle_dtype = paddle.float32
        self.x_type = np.float32
        self.batch = 1
        self.num_head = 8
        self.query_seq_len = 128
        self.kv_seq_len = 128
        self.head_size = 32
        self.bias_type = MaskType.NoMask
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0
        self.max_seqlen_q_ = 0

        self.bias_type = False

        self.seqlen_k = None

        self.causal = True
        self.dropout_p = 0.0
        self.scale = float(1.0 / math.sqrt(self.head_size))

        self.compute_logsumexp = False

    # set query/key/value
    def GetQKV(self, type):
        self.query_static = np.random.uniform(
            low=-0.1,
            high=0.1,
            size=[
                self.batch,
                self.query_seq_len,
                self.num_head,
                self.head_size,
            ],
        ).astype(type)
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
        ).astype(type)
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
        ).astype(type)
        self.value = paddle.to_tensor(
            self.value_static,
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
        ).astype(type)
        self.predect = paddle.to_tensor(
            self.predect_static,
            dtype=self.paddle_dtype,
        )

    # config bias/cu_seqlens_q/cu_seqlens_k/seqstart_q/seqstart_k/
    #       causal_diagonal/seqlen_k/max_seqlen_q/max_seqlen_k
    def GetBias(self, type):

        self.cu_seqlens_q = None
        self.cu_seqlens_k = None

        self.seqstart_q = 0
        self.seqstart_k = 0

        self.causal_diagonal = None
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0
        self.max_seqlen_q_ = 0

        if (self.bias_type) == MaskType.NoMask:
            self.bias = None
        elif (self.bias_type) == MaskType.Random:
            self.bias_static = np.random.uniform(
                low=-0.1,
                high=0.1,
                size=[
                    self.batch,
                    self.num_head,
                    self.query_seq_len,
                    self.kv_seq_len,
                ],
            ).astype(type)
            self.bias = paddle.to_tensor(
                self.bias_static,
                dtype=self.paddle_dtype,
            )
        else:
            assert "We don't approve this type now"

    def GetBaselineOut(self):
        paddle.disable_static()

        self.add_mask = isinstance(self.mask, paddle.Tensor)
        self.use_dropout = self.dropout_p != 0.0

        assert isinstance(self.query, paddle.Tensor)
        query = paddle.transpose(self.query, [0, 2, 1, 3])
        key = paddle.transpose(self.key, [0, 2, 1, 3])
        value = paddle.transpose(self.value, [0, 2, 1, 3])

        qk_res = paddle.matmul(query, key, transpose_y=True)
        attention = qk_res * self.scale
        if self.add_mask:
            attention = attention + self.mask

        softmax_result = paddle.nn.functional.softmax(attention, -1)
        result = paddle.matmul(softmax_result, value)
        if self.use_dropout:
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

    def GetFusedMultiheadAttentionDynOut(self):
        paddle.disable_static()

        print("GetFusedMultiheadAttentionOut program constructing:")

        query = paddle.transpose(self.query, [0, 2, 1, 3])
        key = paddle.transpose(self.key, [0, 2, 1, 3])
        value = paddle.transpose(self.value, [0, 2, 1, 3])

        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False

        if self.bias_type == MaskType.Random:
            (
                output,
                logsumexp,
                seed_and_offset,
            ) = cutlass_fused_multi_head_attention(
                query=query,
                key=key,
                value=value,
                bias=self.bias,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                seqstart_q=None,
                seqstart_k=None,
                causal_diagonal=None,
                seqlen_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                causal=self.causal,
                dropout_p=self.dropout_p,
                scale=self.scale,
                max_seqlen_q_=self.max_seqlen_q_,
                compute_logsumexp=self.compute_logsumexp,
            )
        else:
            (
                output,
                logsumexp,
                seed_and_offset,
            ) = cutlass_fused_multi_head_attention(
                query=query,
                key=key,
                value=value,
                bias=None,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                seqstart_q=None,
                seqstart_k=None,
                causal_diagonal=None,
                seqlen_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                causal=self.causal,
                dropout_p=self.dropout_p,
                scale=self.scale,
                max_seqlen_q_=self.max_seqlen_q_,
                compute_logsumexp=self.compute_logsumexp,
            )

            # import pdb; pdb.set_trace()
            loss = paddle.mean(output - self.predect)
            loss.stop_gradient = False
            loss.backward()

            q_g = query.grad
            k_g = key.grad
            v_g = value.grad

        print("GetFusedMultiheadAttentionOut program has been constructed")

        return output, q_g, k_g, v_g

    def GetFusedMultiheadAttentionStaticOut(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()

        print("GetFusedMultiheadAttentionOut program constructing:")
        with paddle.static.program_guard(main_prog):
            query = paddle.static.data(
                name='query',
                dtype='float32',
                shape=[
                    self.batch,
                    self.query_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            )
            key = paddle.static.data(
                name='key',
                dtype='float32',
                shape=[
                    self.batch,
                    self.kv_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            )
            value = paddle.static.data(
                name='value',
                dtype='float32',
                shape=[
                    self.batch,
                    self.kv_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            )
            predect = paddle.static.data(
                name='predect',
                dtype='float32',
                shape=[
                    self.batch,
                    self.query_seq_len,
                    self.num_head,
                    self.head_size,
                ],
            )

            query.stop_gradient = False
            key.stop_gradient = False
            value.stop_gradient = False

            fetch_list = [
                "output",
                "output_grad",
                "logsumexp",
                "seed_and_offset",
                "query_grad",
                "key_grad",
                "value_grad",
            ]

            feed_list = {
                "query": self.query_static,
                "key": self.key_static,
                "value": self.value_static,
            }

            if self.bias_type == MaskType.Random:
                bias = paddle.static.data(
                    name='bias',
                    dtype='float32',
                    shape=[
                        self.batch,
                        self.num_head,
                        self.query_seq_len,
                        self.kv_seq_len,
                    ],
                )
                feed_list["bias"] = bias

                (
                    output,
                    logsumexp,
                    seed_and_offset,
                ) = cutlass_fused_multi_head_attention(
                    query=query,
                    key=key,
                    value=value,
                    bias=bias,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    seqstart_q=None,
                    seqstart_k=None,
                    causal_diagonal=None,
                    seqlen_k=None,
                    max_seqlen_q=self.max_seqlen_q,
                    max_seqlen_k=self.max_seqlen_k,
                    causal=self.causal,
                    dropout_p=self.dropout_p,
                    scale=self.scale,
                    max_seqlen_q_=self.max_seqlen_q_,
                    compute_logsumexp=self.compute_logsumexp,
                )
            else:
                (
                    output,
                    logsumexp,
                    seed_and_offset,
                ) = cutlass_fused_multi_head_attention(
                    query=query,
                    key=key,
                    value=value,
                    bias=None,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    seqstart_q=None,
                    seqstart_k=None,
                    causal_diagonal=None,
                    seqlen_k=None,
                    max_seqlen_k=self.max_seqlen_k,
                    max_seqlen_q=self.max_seqlen_q,
                    causal=self.causal,
                    dropout_p=self.dropout_p,
                    scale=self.scale,
                    max_seqlen_q_=self.max_seqlen_q_,
                    compute_logsumexp=self.compute_logsumexp,
                )

            # import pdb; pdb.set_trace()
            loss = paddle.mean(output - predect)
            paddle.static.append_backward(loss)

            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            op, op_grad, lse, so, q_g, k_g, v_g = exe.run(
                main_prog, feed=feed_list, fetch_list=fetch_list
            )

        print("GetFusedMultiheadAttentionOut program has been constructed")
        paddle.disable_static()

        return op, op_grad, lse, so, q_g, k_g, v_g

    def test_baseline_op(self):
        self.setUp()
        (
            op,
            op_grad,
            lse,
            so,
            q_g,
            k_g,
            v_g,
        ) = self.GetFusedMultiheadAttentionDynOut()


class TestCutlassFMHAMaskOpDatatype(TestCutlassFMHAOp):
    def config(self):
        super(TestCutlassFMHAMaskOpDatatype).config()
        self.x_type = np.float32
        if self.x_type == np.float16:
            self.paddle_dtype = paddle.float16
            self.x_type = np.float16
        elif self.x_type == np.float32:
            self.paddle_dtype = paddle.float32
            self.x_type = np.float32
        else:
            assert f"this type ({self.x_type}) is not supported"

        self.batch = 1
        self.num_head = 8
        self.query_seq_len = 128
        self.kv_seq_len = 128
        self.head_size = 32
        self.bias_type = MaskType.NoMask
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0
        self.max_seqlen_q_ = 0

        self.seqlen_k = None

        self.causal = True
        self.dropout_p = 0.0
        self.scale = float(1.0 / math.sqrt(self.head_size))

        self.compute_logsumexp = False


if __name__ == "__main__":
    unittest.main()
