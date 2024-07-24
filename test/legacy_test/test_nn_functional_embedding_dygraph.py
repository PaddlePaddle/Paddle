#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.nn.functional.input import embedding_renorm_

paddle.disable_static()


class EmbeddingDygraph(unittest.TestCase):
    def test_1(self):
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)

        embedding = paddle.nn.Embedding(10, 3, sparse=True, padding_idx=9)

        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding.weight.set_value(w0)

        adam = paddle.optimizer.Adam(
            parameters=[embedding.weight], learning_rate=0.01
        )
        adam.clear_grad()

        out = embedding(x)
        out.backward()
        adam.step()

    def test_2(self):
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        y_data = np.arange(6, 12).reshape((3, 2)).astype(np.float32)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)
        y = paddle.to_tensor(y_data, stop_gradient=False)

        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(10, 3, padding_idx=11, sparse=True)

        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(-1, 3, sparse=True)

        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(10, -3, sparse=True)

    def test_3_renorm(self):
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int64)
        weight_np = np.random.random((10, 4)).astype(np.float32) * 10
        max_norm = 5.0
        norm_type = 2.0
        y_ref = self.ref_embedding_renorm_(x, weight_np, max_norm, norm_type)
        weight = paddle.to_tensor(weight_np)
        embedding_renorm_(
            paddle.to_tensor(x),
            weight,
            max_norm,
            norm_type,
        )
        np.testing.assert_allclose(weight.numpy(), y_ref, atol=1e-5)

    def test_4_renorm(self):
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)

        max_norm = 0.5
        norm_type = 3.0
        embedding = paddle.nn.Embedding(
            10,
            3,
            sparse=True,
            padding_idx=9,
            max_norm=max_norm,
            norm_type=norm_type,
        )

        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding.weight.set_value(w0)

        adam = paddle.optimizer.Adam(
            parameters=[embedding.weight], learning_rate=0.01
        )
        adam.clear_grad()

        out = embedding(x)
        out.backward()
        adam.step()

    def ref_embedding_renorm_(self, x, weight, max_norm, norm_type=2.0):
        x = np.reshape(x, (-1,))
        x = np.unique(x)
        x = np.sort(x)
        for i in range(len(x)):
            norm = np.linalg.norm(
                weight[int(x[i])], ord=norm_type, axis=0, keepdims=False
            )
            if norm > max_norm:
                weight[int(x[i])] *= max_norm / (norm + 1e-7)
        return weight


if __name__ == '__main__':
    unittest.main()
