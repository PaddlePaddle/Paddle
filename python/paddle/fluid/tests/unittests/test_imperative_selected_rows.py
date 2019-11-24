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

from __future__ import print_function

import unittest
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.nn import Embedding
import numpy as np


class SimpleNet(fluid.Layer):
    def __init__(self, name_scope, vocab_size, hidden_size):
        super(SimpleNet, self).__init__(name_scope)
        self.emb = fluid.dygraph.Embedding(
            self.full_name(),
            size=[vocab_size, hidden_size],
            dtype='float32',
            param_attr='emb.w',
            is_sparse=True)

    def forward(self, input):
        input_emb = self.emb(input)
        return input_emb, self.emb


class TestSimpleNet(unittest.TestCase):
    def test_selectedrows_gradient(self):
        with fluid.dygraph.guard():
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
            grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(5.0)

            input_word = np.array([[[1], [2]], [[2], [1]]]).astype('int64')
            input = to_variable(input_word)

            simplenet = SimpleNet("SimpleNet", 20, 32)
            input_emb, emb = simplenet(input)

            try:
                emb._w.gradient()
            except ValueError as e:
                pass
            try:
                input_emb.gradient()
            except ValueError as e:
                pass

            input_emb.backward()
            adam.minimize(input_emb, grad_clip=grad_clip)
            emb._w.gradient()

            emb.clear_gradients()
            try:
                emb._w.gradient()
            except ValueError as e:
                pass

            input_emb.clear_gradient()
            try:
                input_emb.gradient()
            except ValueError as e:
                pass


if __name__ == '__main__':
    unittest.main()
