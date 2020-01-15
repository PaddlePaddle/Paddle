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
from paddle.fluid.optimizer import SGDOptimizer
import numpy as np
import paddle.fluid.core as core


class SimpleNet(fluid.Layer):
    def __init__(self, vocab_size, hidden_size, dtype):
        super(SimpleNet, self).__init__()
        self.emb = fluid.dygraph.Embedding(
            size=[vocab_size, hidden_size],
            dtype=dtype,
            param_attr='emb.w',
            is_sparse=True)

    def forward(self, input):
        input_emb = self.emb(input)
        return input_emb, self.emb


class TestSimpleNet(unittest.TestCase):
    def test_selectedrows_gradient1(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    with fluid.dygraph.guard(place):
                        backward_strategy = fluid.dygraph.BackwardStrategy()
                        backward_strategy.sort_sum_gradient = sort_sum_gradient
                        # grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(5.0)

                        input_word = np.array([[1, 2], [2, 1]]).astype('int64')
                        input = to_variable(input_word)

                        simplenet = SimpleNet(20, 32, dtype)
                        adam = SGDOptimizer(
                            learning_rate=0.001,
                            parameter_list=simplenet.parameters())
                        input_emb, emb = simplenet(input)

                        try:
                            emb.weight.gradient()
                        except ValueError as e:
                            assert "has no grad, Please set Variable.stop_gradient=False, or check if this is the first and only variable need grad, if so, please set its pre-Variable's stop_gradient=False, to make sure it has gradient" in str(
                                e)
                        try:
                            input_emb.gradient()
                        except ValueError as e:
                            assert "has no grad, Please set Variable.stop_gradient=False, or check if this is the first and only variable need grad, if so, please set its pre-Variable's stop_gradient=False, to make sure it has gradient" in str(
                                e)

                        input_emb.backward(backward_strategy)
                        adam.minimize(input_emb)  # grad_clip=grad_clip
                        emb.weight.gradient()

                        emb.clear_gradients()
                        try:
                            emb.weight.gradient()
                        except ValueError as e:
                            assert "has no grad, Please set Variable.stop_gradient=False, or check if this is the first and only variable need grad, if so, please set its pre-Variable's stop_gradient=False, to make sure it has gradient" in str(
                                e)

                        input_emb.clear_gradient()
                        input_emb.gradient()

    def test_selectedrows_gradient2(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for sort_sum_gradient in [True, False]:
                with fluid.dygraph.guard(place):
                    backward_strategy = fluid.dygraph.BackwardStrategy()
                    backward_strategy.sort_sum_gradient = sort_sum_gradient
                    grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(
                        5.0)

                    input_word = np.array([[1, 2], [2, 1]]).astype('int64')
                    input = to_variable(input_word)

                    simplenet = SimpleNet(20, 32, "float32")
                    adam = SGDOptimizer(
                        learning_rate=0.001,
                        parameter_list=simplenet.parameters())
                    input_emb, emb = simplenet(input)

                    try:
                        emb.weight.gradient()
                    except ValueError as e:
                        assert "has no grad, Please set Variable.stop_gradient=False, or check if this is the first and only variable need grad, if so, please set its pre-Variable's stop_gradient=False, to make sure it has gradient" in str(
                            e)
                    try:
                        input_emb.gradient()
                    except ValueError as e:
                        assert "has no grad, Please set Variable.stop_gradient=False, or check if this is the first and only variable need grad, if so, please set its pre-Variable's stop_gradient=False, to make sure it has gradient" in str(
                            e)

                    input_emb.backward(backward_strategy)
                    adam.minimize(input_emb, grad_clip=grad_clip)
                    emb.weight.gradient()

                    emb.clear_gradients()
                    try:
                        emb.weight.gradient()
                    except ValueError as e:
                        assert "has no grad, Please set Variable.stop_gradient=False, or check if this is the first and only variable need grad, if so, please set its pre-Variable's stop_gradient=False, to make sure it has gradient" in str(
                            e)

                    input_emb.clear_gradient()
                    input_emb.gradient()


if __name__ == '__main__':
    unittest.main()
