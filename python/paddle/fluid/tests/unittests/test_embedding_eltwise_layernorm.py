#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np
from test_layer_norm_op import _reference_layer_norm_naive


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestEmbeddingEltwiseLayerNorm(OpTest):
    def config(self):
        self.seq_len = 128
        self.size_per_head = 64
        self.head_number = 12
        self.batch_size = 1
        self.scale = 0.125
        self.word_length = 100
        self.input_shape = (1, 128, 1)
        self.begin_norm_axis = 2

    def setUp(self):
        # batch_size, max_sequence_length, hidden dimension
        self.op_type = "fused_embedding_eltwise_layernorm"
        self.config()
        w = self.head_number * self.size_per_head
        self.word = np.random.randint(
            self.word_length, size=self.input_shape).astype("int64")
        self.pos = np.random.randint(
            self.word_length, size=self.input_shape).astype("int64")
        self.sent = np.random.randint(1, size=self.input_shape).astype("int64")
        self.word_w = np.random.random((self.word_length, w)).astype("float32")
        self.pos_w = np.random.random((self.word_length, w)).astype("float32")
        self.sent_w = np.random.random((2, w)).astype("float32")
        word_emb = self.word_w[self.word.flatten(), :]
        pos_emb = self.pos_w[self.pos.flatten(), :]
        sent_emb = self.sent_w[self.sent.flatten(), :]

        scale = np.random.random_sample(w).astype(np.float32)
        bias = np.random.random_sample(w).astype(np.float32)

        eltwise_out = word_emb + pos_emb + sent_emb
        eltwise_out = eltwise_out[np.newaxis, :]
        epsilon = 0.00001
        out, mean, variance = _reference_layer_norm_naive(
            eltwise_out, scale, bias, epsilon, self.begin_norm_axis)

        self.inputs = {
            "WordId": self.word,
            "PosId": self.pos,
            "SentId": self.sent,
            "WordEmb": self.word_w,
            "PosEmb": self.pos_w,
            "SentEmb": self.sent_w,
            "Bias": bias,
            "Scale": scale
        }
        self.attrs = {"epsilon": epsilon, }
        self.outputs = {"Out": out}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
