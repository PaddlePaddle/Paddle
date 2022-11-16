#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core

np.random.random(123)


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Paddle core is not compiled with CUDA"
)
class TestFusedMultiheadMatmulOp(OpTest):
    def config(self):
        self.seq_len = 128
        self.size_per_head = 64
        self.head_number = 12
        self.batch_size = 1
        self.scale = 0.125

    def setUp(self):
        self.op_type = "multihead_matmul"
        self.config()
        h = self.seq_len
        w = self.head_number * self.size_per_head
        self.Input = (
            np.random.random((self.batch_size, h, w)).astype("float32") - 0.5
        )
        self.WQ = np.random.random((w, w)).astype("float32")
        self.KQ = np.random.random((w, w)).astype("float32")
        self.VQ = np.random.random((w, w)).astype("float32")
        self.CombinedW = np.hstack((self.WQ, self.KQ, self.VQ)).reshape(
            (w, 3, w)
        )
        self.Q = np.dot(self.Input, self.WQ)
        self.K = np.dot(self.Input, self.KQ)
        self.V = np.dot(self.Input, self.VQ)

        self.BiasQ = np.random.random((1, w)).astype("float32")
        self.BiasK = np.random.random((1, w)).astype("float32")
        self.BiasV = np.random.random((1, w)).astype("float32")
        self.CombinedB = np.vstack((self.BiasQ, self.BiasK, self.BiasV))
        self.BiasQK = np.random.random(
            (self.batch_size, self.head_number, self.seq_len, self.seq_len)
        ).astype("float32")
        # Compute Q path
        fc_q = self.Q + self.BiasQ
        reshape_q = np.reshape(
            fc_q,
            (
                self.batch_size,
                self.seq_len,
                self.head_number,
                self.size_per_head,
            ),
        )
        transpose_q = np.transpose(reshape_q, (0, 2, 1, 3))
        scale_q = self.scale * transpose_q
        # Compute K path
        fc_k = self.K + self.BiasK
        reshape_k = np.reshape(
            fc_k,
            (
                self.batch_size,
                self.seq_len,
                self.head_number,
                self.size_per_head,
            ),
        )
        transpose_k = np.transpose(reshape_k, (0, 2, 3, 1))

        # Compute Q*K
        q_k = np.matmul(scale_q, transpose_k)
        eltadd_qk = q_k + self.BiasQK
        softmax_qk = np.apply_along_axis(stable_softmax, 3, eltadd_qk)
        # Compute V path
        fc_v = self.V + self.BiasV
        reshape_v = np.reshape(
            fc_v,
            (
                self.batch_size,
                self.seq_len,
                self.head_number,
                self.size_per_head,
            ),
        )
        transpose_v = np.transpose(reshape_v, (0, 2, 1, 3))

        # Compute QK*V
        qkv = np.matmul(softmax_qk, transpose_v)
        transpose_qkv = np.transpose(qkv, (0, 2, 1, 3))
        reshape_qkv = np.reshape(transpose_qkv, (self.batch_size, h, w))

        self.inputs = {
            "Input": self.Input,
            "W": self.CombinedW,
            "Bias": self.CombinedB,
            "BiasQK": self.BiasQK,
        }
        self.attrs = {
            "transpose_Q": False,
            "transpose_K": True,
            "transpose_V": False,
            "head_number": self.head_number,
            "alpha": self.scale,
        }
        self.outputs = {"Out": reshape_qkv}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=2e-3)


class TestFusedMultiHeadMatmulOp2(TestFusedMultiheadMatmulOp):
    def config(self):
        self.seq_len = 256
        self.size_per_head = 32
        self.head_number = 12
        self.batch_size = 8
        self.scale = 0.125

# unit test for biasQK with shape (1,1,seqlen,seqlen)
class TestFusedMultiHeadMatmulOp_biasqk2(OpTest):
    def config(self):
        self.seq_len = 128
        self.size_per_head = 64
        self.head_number = 12
        self.batch_size = 1
        self.scale = 0.125

    def setUp(self):
        self.op_type = "multihead_matmul"
        self.config()
        h = self.seq_len
        w = self.head_number * self.size_per_head
        self.Input = (
            np.random.random((self.batch_size, h, w)).astype("float32") - 0.5
        )
        self.WQ = np.random.random((w, w)).astype("float32")
        self.KQ = np.random.random((w, w)).astype("float32")
        self.VQ = np.random.random((w, w)).astype("float32")
        self.CombinedW = np.hstack((self.WQ, self.KQ, self.VQ)).reshape(
            (w, 3, w)
        )
        self.Q = np.dot(self.Input, self.WQ)
        self.K = np.dot(self.Input, self.KQ)
        self.V = np.dot(self.Input, self.VQ)

        self.BiasQ = np.random.random((1, w)).astype("float32")
        self.BiasK = np.random.random((1, w)).astype("float32")
        self.BiasV = np.random.random((1, w)).astype("float32")
        self.CombinedB = np.vstack((self.BiasQ, self.BiasK, self.BiasV))
        self.BiasQK = np.random.random(
            (1, 1, self.seq_len, self.seq_len)
        ).astype("float32")
        # Compute Q path
        fc_q = self.Q + self.BiasQ
        reshape_q = np.reshape(
            fc_q,
            (
                self.batch_size,
                self.seq_len,
                self.head_number,
                self.size_per_head,
            ),
        )
        transpose_q = np.transpose(reshape_q, (0, 2, 1, 3))
        scale_q = self.scale * transpose_q
        # Compute K path
        fc_k = self.K + self.BiasK
        reshape_k = np.reshape(
            fc_k,
            (
                self.batch_size,
                self.seq_len,
                self.head_number,
                self.size_per_head,
            ),
        )
        transpose_k = np.transpose(reshape_k, (0, 2, 3, 1))

        # Compute Q*K
        q_k = np.matmul(scale_q, transpose_k)
        eltadd_qk = q_k + self.BiasQK
        softmax_qk = np.apply_along_axis(stable_softmax, 3, eltadd_qk)
        # Compute V path
        fc_v = self.V + self.BiasV
        reshape_v = np.reshape(
            fc_v,
            (
                self.batch_size,
                self.seq_len,
                self.head_number,
                self.size_per_head,
            ),
        )
        transpose_v = np.transpose(reshape_v, (0, 2, 1, 3))

        # Compute QK*V
        qkv = np.matmul(softmax_qk, transpose_v)
        transpose_qkv = np.transpose(qkv, (0, 2, 1, 3))
        reshape_qkv = np.reshape(transpose_qkv, (self.batch_size, h, w))

        self.inputs = {
            "Input": self.Input,
            "W": self.CombinedW,
            "Bias": self.CombinedB,
            "BiasQK": self.BiasQK,
        }
        self.attrs = {
            "transpose_Q": False,
            "transpose_K": True,
            "transpose_V": False,
            "head_number": self.head_number,
            "alpha": self.scale,
        }
        self.outputs = {"Out": reshape_qkv}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=2e-3)


if __name__ == '__main__':
    unittest.main()
