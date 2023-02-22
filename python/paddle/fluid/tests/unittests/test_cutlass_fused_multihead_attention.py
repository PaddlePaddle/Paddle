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
import unittest

import numpy as np
from op_test import OpTest

# Ensure we use float type to accumulate
os.environ["FLAGS_gemm_use_half_precision_compute_type"] = "0"

import paddle
from paddle.fluid.framework import default_main_program
from paddle.incubate.nn.functional import cutlass_fused_multi_head_attention

default_main_program().random_seed = 42


class TestCutlassFMHAOp(OpTest):
    def setUp(self):
        self.config()
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
            dtype=paddle.float16,
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
            dtype=paddle.float16,
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
            dtype=paddle.float16,
        )
        self.mask = paddle.to_tensor(
            np.random.uniform(low=-0.01, high=0.01, size=self.mask_shape),
            dtype=paddle.float16,
        )

        self.query.stop_gradient = True
        self.key.stop_gradient = True
        self.value.stop_gradient = True
        paddle.set_default_dtype("float16")

    def config(self):
        self.x_type = np.float16
        self.batch = 1
        self.num_head = 8
        self.query_seq_len = 900
        self.kv_seq_len = 6000
        self.head_size = 32
        self.scale = float(1.0 / math.sqrt(self.head_size))
        self.add_mask = True
        self.mask_shape = [
            self.batch,
            self.num_head,
            self.query_seq_len,
            self.kv_seq_len,
        ]

    def GetBaselineOut(self):
        paddle.disable_static()
        # Transpose to (batch, num_head, seq_len, head_size)
        query = paddle.transpose(self.query, [0, 2, 1, 3])
        key = paddle.transpose(self.key, [0, 2, 1, 3])
        value = paddle.transpose(self.value, [0, 2, 1, 3])

        qk_res = paddle.matmul(query, key, transpose_y=True)
        attention = qk_res * self.scale
        if self.add_mask:
            attention = attention + self.mask

        softmax_result = paddle.nn.functional.softmax(attention, -1)
        result = paddle.matmul(softmax_result, value)
        result = paddle.transpose(result, [0, 2, 1, 3])
        return result

    def GetFusedMultiheadAttentionOut(self):
        paddle.disable_static()
        fused_out = cutlass_fused_multi_head_attention(
            self.query,
            self.key,
            self.value,
            self.mask if self.add_mask else None,
            self.scale,
        )

        return fused_out

    def test_fused_multihead_attention_op(self):
        final_out_ref = self.GetBaselineOut()
        final_out = self.GetFusedMultiheadAttentionOut()

        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=self.rtol, atol=self.atol
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
        self.mask_shape = [
            self.batch,
            self.num_head,
            self.query_seq_len,
            self.kv_seq_len,
        ]


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
        self.mask_shape = [1, 1, 1, self.kv_seq_len]


class TestCutlassFMHAOpFp32(TestCutlassFMHAOp):
    def config(self):
        super().config()
        self.x_type = np.float16
        self.batch = 8
        self.num_head = 8
        self.query_seq_len = 128
        self.kv_seq_len = 128
        self.head_size = 32
        self.scale = float(1.0 / math.sqrt(self.head_size))
        self.add_mask = True
        self.mask_shape = [
            self.batch,
            self.num_head,
            self.query_seq_len,
            self.kv_seq_len,
        ]


if __name__ == "__main__":
    unittest.main()
