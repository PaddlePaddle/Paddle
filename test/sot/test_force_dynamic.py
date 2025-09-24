# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit import sot


class EmbeddingLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = paddle.nn.Embedding(10, 10)

    def forward(self, x):
        x = x + 1 - 1
        x = self.embedding(x)
        return x + 1


def call_embedding_layer(x: paddle.Tensor, layer: paddle.nn.Layer):
    return layer(x)


def call_functional_embedding(x: paddle.Tensor, weight: paddle.Tensor):
    x = x + 1 - 1
    x = paddle.nn.functional.embedding(x, weight)
    return x + 1


class TestForceDynamic(TestCaseBase):
    def test_embedding_layer(self):
        paddle.jit.marker.force_dynamic(paddle.nn.Embedding)

        layer = EmbeddingLayer()
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(
                call_embedding_layer,
                paddle.randint(0, 10, [1, 3, 224, 224], dtype='int64'),
                layer,
            )
            self.assertGreater(ctx.translate_count, 1)

        sot.utils.paddle_api_config.break_graph_layer_classes.clear()

    def test_functional_embedding(self):
        paddle.jit.marker.force_dynamic(paddle.nn.functional.embedding)

        weight = paddle.randn([10, 10])
        x = paddle.randint(0, 10, [1, 3, 224, 224], dtype='int64')
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(call_functional_embedding, x, weight)
            self.assertGreater(ctx.translate_count, 1)

        sot.utils.paddle_api_config.break_graph_functions.clear()


if __name__ == "__main__":
    unittest.main()
