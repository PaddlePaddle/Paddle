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
from paddle import base
from paddle.nn import functional


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

                emb = functional.embedding(
                    x=label, weight=weight, sparse=True, name="embedding"
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

                emb = functional.embedding(
                    x=label,
                    weight=weight,
                    padding_idx=129,
                    sparse=True,
                    name="embedding",
                )

        with self.assertRaises(ValueError):
            test_bad_x()


if __name__ == '__main__':
    unittest.main()
