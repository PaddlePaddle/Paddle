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

import unittest

import numpy as np
from test_sparse_attention_op import get_cuda_version

import paddle
import paddle.nn.functional as F


class BF16EmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 30
        self.vocab_size = 1024
        self.hidden_size = 512
        self.seed = 10

    def run_main(self, dtype):
        ids, weight, dout = self.gen_random()
        origin_dtype = weight.dtype
        weight_cast = weight.astype(dtype)
        out = F.embedding(ids, weight_cast)
        dout = dout.astype(out.dtype)
        dweight = paddle.autograd.grad(out, weight, dout)
        return (
            out.astype(origin_dtype).numpy(),
            dweight[0].astype(origin_dtype).numpy(),
        )

    def gen_random(self):
        np.random.seed(self.seed)
        weight = np.random.random([self.vocab_size, self.hidden_size]).astype(
            'float32'
        )
        ids = np.random.randint(
            low=0, high=self.vocab_size, size=[self.batch_size]
        )
        dout = np.random.random([self.batch_size, self.hidden_size]).astype(
            'float32'
        )

        weight = paddle.to_tensor(weight)
        weight.stop_gradient = False
        ids = paddle.to_tensor(ids)
        dout = paddle.to_tensor(dout)
        return ids, weight, dout

    def test_main(self):
        if not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000:
            return

        ret1 = self.run_main('float32')
        ret2 = self.run_main('bfloat16')
        self.assertEqual(len(ret1), len(ret2))
        for i, (r1, r2) in enumerate(zip(ret1, ret2)):
            np.testing.assert_allclose(r1, r2, atol=1e-3, rtol=1e-2)


class BF16EmbeddingTestOddHiddenSize(BF16EmbeddingTest):
    def setUp(self):
        self.batch_size = 30
        self.vocab_size = 511
        self.hidden_size = 512
        self.seed = 20


if __name__ == "__main__":
    unittest.main()
