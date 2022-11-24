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

import unittest
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.optimizer import SGDOptimizer
import numpy as np
import paddle.fluid.core as core
import paddle
from paddle.fluid.framework import _test_eager_guard


class SimpleNet(paddle.nn.Layer):

    def __init__(self, vocab_size, hidden_size, dtype):
<<<<<<< HEAD
        super(SimpleNet, self).__init__()
        self.emb = fluid.dygraph.Embedding(size=[vocab_size, hidden_size],
                                           dtype=dtype,
                                           param_attr='emb.w',
                                           is_sparse=True)
=======
        super().__init__()
        self.emb = fluid.dygraph.Embedding(
            size=[vocab_size, hidden_size],
            dtype=dtype,
            param_attr='emb.w',
            is_sparse=True,
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    def forward(self, input):
        input_emb = self.emb(input)
        return input_emb, self.emb


class TestSimpleNet(unittest.TestCase):

    def func_selectedrows_gradient1(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    paddle.disable_static(place)
                    fluid.set_flags(
<<<<<<< HEAD
                        {'FLAGS_sort_sum_gradient': sort_sum_gradient})
=======
                        {'FLAGS_sort_sum_gradient': sort_sum_gradient}
                    )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                    # grad_clip = fluid.clip.GradientClipByGlobalNorm(5.0)

                    input_word = np.array([[1, 2], [2, 1]]).astype('int64')
                    input = paddle.to_tensor(input_word)

                    simplenet = SimpleNet(20, 32, dtype)
<<<<<<< HEAD
                    adam = SGDOptimizer(learning_rate=0.001,
                                        parameter_list=simplenet.parameters()
                                        )  # grad_clip=grad_clip
=======
                    adam = SGDOptimizer(
                        learning_rate=0.001,
                        parameter_list=simplenet.parameters(),
                    )  # grad_clip=grad_clip
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                    input_emb, emb = simplenet(input)

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

    def test_selectedrows_gradient1(self):
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        with _test_eager_guard():
            self.func_selectedrows_gradient1()
        self.func_selectedrows_gradient1()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def func_selectedrows_gradient2(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for sort_sum_gradient in [True, False]:
                with fluid.dygraph.guard(place):
                    fluid.set_flags(
<<<<<<< HEAD
                        {'FLAGS_sort_sum_gradient': sort_sum_gradient})
=======
                        {'FLAGS_sort_sum_gradient': sort_sum_gradient}
                    )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                    grad_clip = fluid.clip.GradientClipByGlobalNorm(5.0)

                    input_word = np.array([[1, 2], [2, 1]]).astype('int64')
                    input = to_variable(input_word)

                    simplenet = SimpleNet(20, 32, "float32")
<<<<<<< HEAD
                    adam = SGDOptimizer(learning_rate=0.001,
                                        parameter_list=simplenet.parameters(),
                                        grad_clip=grad_clip)
=======
                    adam = SGDOptimizer(
                        learning_rate=0.001,
                        parameter_list=simplenet.parameters(),
                        grad_clip=grad_clip,
                    )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                    input_emb, emb = simplenet(input)

                    self.assertIsNone(emb.weight.gradient())
                    self.assertIsNone(input_emb.gradient())

                    input_emb.backward()
                    adam.minimize(input_emb)
                    self.assertIsNotNone(emb.weight.gradient())

                    emb.clear_gradients()
                    self.assertIsNone(emb.weight.gradient())

                    input_emb.clear_gradient()
                    self.assertIsNotNone(input_emb.gradient())

    def test_selectedrows_gradient2(self):
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        with _test_eager_guard():
            self.func_selectedrows_gradient2()
        self.func_selectedrows_gradient2()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


if __name__ == '__main__':
    unittest.main()
