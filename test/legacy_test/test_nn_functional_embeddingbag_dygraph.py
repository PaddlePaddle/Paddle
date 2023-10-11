# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

paddle.disable_static()


class EmbeddingBagDygraph(unittest.TestCase):
    def func_1(self):
        paddle.disable_static(paddle.CPUPlace())

        indices_data = np.random.randint(low=0, high=10, size=(3, 2)).astype(
            np.int64
        )
        indices = paddle.to_tensor(indices_data, stop_gradient=False)

        weight_data = np.random.randint(low=0, high=10, size=(3, 2)).astype(
            np.float32
        )
        weight = paddle.to_tensor(weight_data, stop_gradient=False)

        embedding_bag = paddle.nn.EmbeddingBag(10, 3, mode='sum')

        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding_bag._embedding.set_value(w0)

        adam = paddle.optimizer.Adam(
            parameters=[embedding_bag._embedding], learning_rate=0.01
        )
        adam.clear_grad()

        out = embedding_bag(input=indices, weight=weight)
        out.backward()
        adam.step()

    def test_1(self):
        self.func_1()

    def func_2(self):
        paddle.disable_static(paddle.CPUPlace())

        indices_data = np.random.randint(low=0, high=10, size=(3, 2)).astype(
            np.int64
        )
        indices = paddle.to_tensor(indices_data, stop_gradient=False)

        weight_data = np.random.randint(low=0, high=10, size=(3, 2)).astype(
            np.float32
        )
        weight = paddle.to_tensor(weight_data, stop_gradient=False)

        embedding_bag = paddle.nn.EmbeddingBag(10, 3, mode='mean')

        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding_bag._embedding.set_value(w0)

        adam = paddle.optimizer.Adam(
            parameters=[embedding_bag._embedding], learning_rate=0.01
        )
        adam.clear_grad()

        out = embedding_bag(input=indices, weight=None)
        out.backward()
        adam.step()

    def test_2(self):
        self.func_2()

    def func_3(self):
        paddle.disable_static(paddle.CPUPlace())

        indices_data = np.random.randint(low=0, high=10, size=(3, 2)).astype(
            np.int64
        )
        indices = paddle.to_tensor(indices_data, stop_gradient=False)

        weight_data = np.random.randint(low=0, high=10, size=(3, 2)).astype(
            np.float32
        )
        weight = paddle.to_tensor(weight_data, stop_gradient=False)

        with self.assertRaises(ValueError):
            embedding_bag = paddle.nn.EmbeddingBag(0, 3, mode='mean')

        with self.assertRaises(ValueError):
            embedding_bag = paddle.nn.EmbeddingBag(10, -1, mode='mean')

        with self.assertRaises(ValueError):
            embedding_bag = paddle.nn.EmbeddingBag(10, 3, mode='mean')
            w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
            embedding_bag._embedding.set_value(w0)
            adam = paddle.optimizer.Adam(
                parameters=[embedding_bag._embedding], learning_rate=0.01
            )
            adam.clear_grad()
            out = embedding_bag(input=indices, weight=weight)
            out.backward()
            adam.step()

    def test_3(self):
        self.func_3()


if __name__ == '__main__':
    unittest.main()
