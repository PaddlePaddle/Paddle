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

import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core


class SimpleNet(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size, dtype):
        super().__init__()
        self.emb = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr='emb.w',
            sparse=True,
        )

    def forward(self, input):
        input_emb = self.emb(input)
        return input_emb, self.emb


class TestSimpleNet(unittest.TestCase):
    def test_selectedrows_gradient1(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    paddle.disable_static(place)
                    base.set_flags(
                        {'FLAGS_sort_sum_gradient': sort_sum_gradient}
                    )
                    # grad_clip = paddle.nn.ClipGradByGlobalNorm(5.0)

                    input_word = np.array([[1, 2], [2, 1]]).astype('int64')
                    input = paddle.to_tensor(input_word)

                    simplenet = SimpleNet(20, 32, dtype)
                    adam = paddle.optimizer.SGD(
                        learning_rate=0.001,
                        parameters=simplenet.parameters(),
                    )  # grad_clip=grad_clip
                    input_emb, emb = simplenet(input)
                    input_emb.retain_grads()

                    self.assertIsNone(emb.weight.gradient())
                    self.assertIsNone(input_emb.gradient())

                    input_emb.backward()
                    adam.minimize(input_emb)
                    self.assertIsNotNone(emb.weight.gradient())

                    emb.clear_gradients()
                    self.assertIsNone(emb.weight.gradient())

                    input_emb.clear_gradient()
                    self.assertIsNotNone(input_emb.gradient())
                    paddle.enable_static()

    def test_selectedrows_gradient2(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for place in places:
            for sort_sum_gradient in [True, False]:
                with base.dygraph.guard(place):
                    base.set_flags(
                        {'FLAGS_sort_sum_gradient': sort_sum_gradient}
                    )
                    grad_clip = paddle.nn.ClipGradByGlobalNorm(5.0)

                    input_word = np.array([[1, 2], [2, 1]]).astype('int64')
                    input = paddle.to_tensor(input_word)

                    simplenet = SimpleNet(20, 32, "float32")
                    adam = paddle.optimizer.SGD(
                        learning_rate=0.001,
                        parameters=simplenet.parameters(),
                        grad_clip=grad_clip,
                    )
                    input_emb, emb = simplenet(input)
                    input_emb.retain_grads()

                    self.assertIsNone(emb.weight.gradient())
                    self.assertIsNone(input_emb.gradient())

                    input_emb.backward()
                    adam.minimize(input_emb)
                    self.assertIsNotNone(emb.weight.gradient())

                    emb.clear_gradients()
                    self.assertIsNone(emb.weight.gradient())

                    input_emb.clear_gradient()
                    self.assertIsNotNone(input_emb.gradient())


if __name__ == '__main__':
    unittest.main()
