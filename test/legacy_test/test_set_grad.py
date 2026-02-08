# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestSetGradNone(unittest.TestCase):
    def test_clear_dense_grad(self):
        x = paddle.randn([4, 4], dtype='float32')
        x.stop_gradient = False
        y = x.sum()
        y.backward()
        assert x.grad is not None
        x.grad = None
        assert x.grad is None

    def test_clear_selectedrows_grad(self):
        sparse_emb = paddle.nn.Embedding(
            num_embeddings=100, embedding_dim=16, sparse=True
        )

        x = paddle.randint(0, 100, shape=[10], dtype="int64")
        out = sparse_emb(x).sum()
        out.backward()

        grad = sparse_emb.weight.grad
        self.assertIsNotNone(grad)

        sparse_emb.weight.grad = None

        self.assertIsNone(sparse_emb.weight.grad)


if __name__ == '__main__':
    unittest.main()
