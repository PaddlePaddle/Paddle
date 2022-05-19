#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from op_test import OpTest
from paddle.fluid import core
import paddle.fluid as fluid

np.random.random(123)


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedVitAttentionOp(OpTest):
    def config(self):
        self.seq_len = 128
        self.size_per_head = 64
        self.head_number = 12
        self.batch_size = 1
        self.scale = 1.25

    def setUp(self):
        self.op_type = "vit_attention"
        self.config()
        h = self.seq_len
        hidden_size = self.head_number * self.size_per_head
        # W = QKV
        w = hidden_size * 3
        self.Input = np.random.random(
            (self.batch_size, h, w)).astype("float32") - 0.5
        # reshape
        reshape_input = np.reshape(self.Input,
                                   (self.batch_size, self.seq_len, 3,
                                    self.head_number, self.size_per_head))
        # transpose
        transpose_input = np.transpose(reshape_input, (2, 0, 3, 1, 4))
        # slice 
        self.Q = transpose_input[0, :, :, :, :]
        self.K = transpose_input[1, :, :, :, :]
        self.V = transpose_input[2, :, :, :, :]
        transpose_k = np.transpose(self.K, (0, 1, 3, 2))

        # Compute Q*K
        q_k = np.matmul(self.Q, transpose_k)
        q_k = q_k * self.scale
        softmax_qk = np.apply_along_axis(stable_softmax, 3, q_k)

        #transpose_qk = np.transpose(softmax_qk, (0, 2, 1, 3))
        #reshape_qk = np.reshape(transpose_qk, (self.batch_size, self.seq_len, self.seq_len*self.head_number)) 

        # Compute QK*V
        qkv = np.matmul(softmax_qk, self.V)
        transpose_qkv = np.transpose(qkv, (0, 2, 1, 3))
        reshape_qkv = np.reshape(transpose_qkv,
                                 (self.batch_size, h, hidden_size))
        self.inputs = {"Input": self.Input}
        self.attrs = {"head_number": self.head_number, "scale": self.scale}
        self.outputs = {"Out": reshape_qkv}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=2e-3)


if __name__ == '__main__':
    unittest.main()
