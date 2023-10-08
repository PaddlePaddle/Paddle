#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base
from paddle.nn import functional


class EmbeddingDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_1(self):
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)

        embedding_bag = paddle.nn.embedding_bag(
            10, 3, sparse=True, padding_idx=9
        )

        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding_bag.weight.set_value(w0)

        adam = paddle.optimizer.Adam(
            parameters=[embedding_bag.weight], learning_rate=0.01
        )
        adam.clear_grad()

        out = embedding_bag(x)
        out.backward()
        adam.step()

    def test_2(self):
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        y_data = np.arange(6, 12).reshape((3, 2)).astype(np.float32)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)
        y = paddle.to_tensor(y_data, stop_gradient=False)

        with self.assertRaises(ValueError):
            embedding_bag = paddle.nn.embedding_bag(
                10, 3, padding_idx=11, sparse=True
            )

        with self.assertRaises(ValueError):
            embedding_bag = paddle.nn.embedding_bag(-1, 3, sparse=True)

        with self.assertRaises(ValueError):
            embedding_bag = paddle.nn.embedding_bag(10, -3, sparse=True)


class EmbeddingStatic(unittest.TestCase):
    def test_1(self):
        prog = base.Program()
        with base.program_guard(prog):

            def test_bad_x():
                initializer = paddle.nn.initializer.Assign(
                    np.random.random(size=(128, 100))
                )

                param_attr = base.ParamAttr(
                    name="emb_weight",
                    learning_rate=0.5,
                    initializer=initializer,
                    trainable=True,
                )

                weight = prog.global_block().create_parameter(
                    (128, 100), attr=param_attr, dtype="float32"
                )

                label = paddle.static.data(
                    name="label",
                    shape=[-1, 4],
                    dtype="int64",
                )

                emb = functional.embedding_bag(
                    x=label, weight=weight, sparse=True, name="embedding_bag"
                )

            test_bad_x()

    def test_2(self):
        prog = base.Program()
        with base.program_guard(prog):

            def test_bad_x():
                initializer = paddle.nn.initializer.Assign(
                    np.random.random(size=(128, 100))
                )

                param_attr = base.ParamAttr(
                    name="emb_weight",
                    learning_rate=0.5,
                    initializer=initializer,
                    trainable=True,
                )

                weight = prog.global_block().create_parameter(
                    (128, 100), attr=param_attr, dtype="float32"
                )

                label = paddle.static.data(
                    name="label",
                    shape=[-1, 4],
                    dtype="int32",
                )

                emb = functional.embedding_bag(
                    x=label,
                    weight=weight,
                    padding_idx=129,
                    sparse=True,
                    name="embedding_bag",
                )

        with self.assertRaises(ValueError):
            test_bad_x()


if __name__ == '__main__':
    unittest.main()
