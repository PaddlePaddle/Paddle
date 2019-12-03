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
import paddle.fluid.dygraph.base as base
import paddle.fluid.core as core
import numpy as np

import test_imperative_lod_tensor_to_selected_rows
import test_imperative_selected_rows_to_lod_tensor


class SimpleNetOnlyEmbedding(fluid.Layer):
    def __init__(self, name_scope, vocab_size, hidden_size, dtype, is_sparse):
        super(SimpleNetOnlyEmbedding, self).__init__(name_scope)
        self.emb = fluid.dygraph.Embedding(
            self.full_name(),
            size=[vocab_size, hidden_size],
            dtype=dtype,
            param_attr='emb.w',
            is_sparse=is_sparse)

    def forward(self, input):
        input_emb = self.emb(input)
        return input_emb, self.emb


class SimpleNetSharedEmbedding(fluid.Layer):
    def __init__(self, name_scope, vocab_size, hidden_size, dtype, is_sparse):
        super(SimpleNetSharedEmbedding, self).__init__(name_scope)
        self.emb1 = fluid.dygraph.Embedding(
            self.full_name(),
            size=[vocab_size, hidden_size],
            dtype=dtype,
            param_attr='emb.w1',
            is_sparse=is_sparse)
        self.emb2 = fluid.dygraph.Embedding(
            self.full_name(),
            size=[vocab_size, hidden_size],
            dtype=dtype,
            param_attr='emb.w2',
            is_sparse=is_sparse)
        self.emb2._w = self.emb1._w

    def forward(self, input):
        input_emb1 = self.emb1(input)
        input_emb2 = self.emb2(input)
        input_emb = input_emb1 + input_emb2
        return input_emb, self.emb1


class Test_Grad_Accumulate(unittest.TestCase):
    def test_lod_tensor_to_selected_rows_gradient_accumulate(self):
        seed = 90

        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    for is_sparse in [True, False]:
                        emb_gradient_multi_mini_batch = 0
                        emb_gradient_one_large_batch = 0
                        with fluid.dygraph.guard(place):
                            fluid.default_startup_program().random_seed = seed
                            fluid.default_main_program().random_seed = seed
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array(
                                [0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(
                                    3, 3).astype('int64')
                            input_word = input_word.reshape((-1, 3, 1))
                            y_data = np.arange(1, 10).reshape(3,
                                                              3).astype('int64')
                            y_data = y_data.reshape((-1, 1))

                            input = base.to_variable(input_word)
                            y = base.to_variable(y_data)
                            simplenet = test_imperative_lod_tensor_to_selected_rows.SimpleNet(
                                "SimpleNet",
                                hidden_size=20,
                                vocab_size=32,
                                num_steps=3,
                                init_scale=0.1,
                                is_sparse=is_sparse,
                                dtype=dtype)
                            outs = simplenet(input, y)
                            outs = outs / 2
                            outs.backward(backward_strategy)
                            emb_gradient = simplenet.embedding._w.gradient()

                            outs = simplenet(input, y)
                            outs = outs / 2
                            outs.backward(backward_strategy)

                            emb_gradient_multi_mini_batch = simplenet.embedding._w.gradient(
                            )

                        with fluid.dygraph.guard(place):
                            fluid.default_startup_program().random_seed = seed
                            fluid.default_main_program().random_seed = seed
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array([
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
                                7, 8
                            ]).reshape(6, 3).astype('int64')
                            input_word = input_word.reshape((-1, 3, 1))
                            y_data = np.array([
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7,
                                8, 9
                            ]).reshape(6, 3).astype('int64')
                            y_data = y_data.reshape((-1, 1))

                            input = base.to_variable(input_word)
                            y = base.to_variable(y_data)
                            simplenet = test_imperative_lod_tensor_to_selected_rows.SimpleNet(
                                "SimpleNet",
                                hidden_size=20,
                                vocab_size=32,
                                num_steps=3,
                                init_scale=0.1,
                                is_sparse=is_sparse,
                                dtype=dtype)
                            outs = simplenet(input, y)
                            outs.backward(backward_strategy)
                            emb_gradient_one_large_batch = simplenet.embedding._w.gradient(
                            )

                        self.assertTrue(
                            np.allclose(emb_gradient_multi_mini_batch,
                                        emb_gradient_one_large_batch))

    def test_selected_rows_to_lod_tensor_gradient_accumulate(self):
        seed = 90

        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    for is_sparse in [True, False]:
                        emb_gradient_multi_mini_batch = 0
                        emb_gradient_one_large_batch = 0
                        with fluid.dygraph.guard(place):
                            fluid.default_startup_program().random_seed = seed
                            fluid.default_main_program().random_seed = seed
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array(
                                [0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(
                                    3, 3).astype('int64')
                            input_word = input_word.reshape((-1, 3, 1))
                            y_data = np.arange(1, 10).reshape(3,
                                                              3).astype('int64')
                            y_data = y_data.reshape((-1, 1))

                            input = base.to_variable(input_word)
                            y = base.to_variable(y_data)
                            simplenet = test_imperative_selected_rows_to_lod_tensor.SimpleNet(
                                "SimpleNet",
                                hidden_size=20,
                                vocab_size=32,
                                num_steps=3,
                                init_scale=0.1,
                                is_sparse=is_sparse,
                                dtype=dtype)
                            outs = simplenet(input, y)
                            outs = outs / 2
                            outs.backward(backward_strategy)
                            emb_gradient = simplenet.embedding._w.gradient()

                            outs = simplenet(input, y)
                            outs = outs / 2
                            outs.backward(backward_strategy)

                            emb_gradient_multi_mini_batch = simplenet.embedding._w.gradient(
                            )

                        with fluid.dygraph.guard(place):
                            fluid.default_startup_program().random_seed = seed
                            fluid.default_main_program().random_seed = seed
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array([
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
                                7, 8
                            ]).reshape(6, 3).astype('int64')
                            input_word = input_word.reshape((-1, 3, 1))
                            y_data = np.array([
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7,
                                8, 9
                            ]).reshape(6, 3).astype('int64')
                            y_data = y_data.reshape((-1, 1))

                            input = base.to_variable(input_word)
                            y = base.to_variable(y_data)
                            simplenet = test_imperative_selected_rows_to_lod_tensor.SimpleNet(
                                "SimpleNet",
                                hidden_size=20,
                                vocab_size=32,
                                num_steps=3,
                                init_scale=0.1,
                                is_sparse=is_sparse,
                                dtype=dtype)
                            outs = simplenet(input, y)
                            outs.backward(backward_strategy)
                            emb_gradient_one_large_batch = simplenet.embedding._w.gradient(
                            )

                        self.assertTrue(
                            np.allclose(emb_gradient_multi_mini_batch,
                                        emb_gradient_one_large_batch))

    def test_simplenet_shared_embedding_gradient_accumulate(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    for is_sparse in [True, False]:
                        with fluid.dygraph.guard(place):
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array(
                                [[[1], [2]], [[2], [1]]]).astype('int64')
                            input = base.to_variable(input_word)
                            simplenet = SimpleNetSharedEmbedding(
                                "SimpleNetSharedEmbedding", 20, 32, dtype,
                                is_sparse)
                            input_emb, emb = simplenet(input)
                            input_emb.backward(backward_strategy)
                            adam.minimize(input_emb)
                            emb_gradient = emb._w.gradient()

                            input_emb, emb = simplenet(input)
                            input_emb.backward(backward_strategy)

                            if is_sparse:
                                self.assertTrue(
                                    np.array_equal(emb._w.gradient()[0],
                                                   emb_gradient[0] * 2))
                            else:
                                self.assertTrue(
                                    np.array_equal(emb._w.gradient(),
                                                   emb_gradient * 2))

                            simplenet.clear_gradients()

    def test_simplenet_only_embedding_gradient_accumulate(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    for is_sparse in [True, False]:
                        with fluid.dygraph.guard(place):
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array(
                                [[[1], [2]], [[2], [1]]]).astype('int64')
                            input = base.to_variable(input_word)
                            simplenet = SimpleNetOnlyEmbedding(
                                "SimpleNetOnlyEmbedding", 20, 32, dtype,
                                is_sparse)
                            input_emb, emb = simplenet(input)
                            input_emb.backward(backward_strategy)
                            adam.minimize(input_emb)
                            emb_gradient = emb._w.gradient()

                            input_emb, emb = simplenet(input)
                            input_emb.backward(backward_strategy)

                            if is_sparse:
                                self.assertTrue(
                                    np.array_equal(emb._w.gradient()[0],
                                                   emb_gradient[0] * 2))
                            else:
                                self.assertTrue(
                                    np.array_equal(emb._w.gradient(),
                                                   emb_gradient * 2))

                            simplenet.clear_gradients()

    def test_selected_rows_to_lod_tensor_clear_gradient(self):
        seed = 90

        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    for is_sparse in [True, False]:
                        emb_gradient_multi_mini_batch = 0
                        emb_gradient_one_large_batch = 0
                        with fluid.dygraph.guard(place):
                            fluid.default_startup_program().random_seed = seed
                            fluid.default_main_program().random_seed = seed
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array(
                                [0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(
                                    3, 3).astype('int64')
                            input_word = input_word.reshape((-1, 3, 1))
                            y_data = np.arange(1, 10).reshape(3,
                                                              3).astype('int64')
                            y_data = y_data.reshape((-1, 1))

                            input = base.to_variable(input_word)
                            y = base.to_variable(y_data)
                            simplenet = test_imperative_selected_rows_to_lod_tensor.SimpleNet(
                                "SimpleNet",
                                hidden_size=20,
                                vocab_size=32,
                                num_steps=3,
                                init_scale=0.1,
                                is_sparse=is_sparse,
                                dtype=dtype)
                            outs = simplenet(input, y)
                            outs.backward(backward_strategy)
                            emb_gradient = simplenet.embedding._w.gradient()
                            simplenet.clear_gradients()

                            outs = simplenet(input, y)
                            outs.backward(backward_strategy)

                            self.assertTrue(
                                np.array_equal(simplenet.embedding._w.gradient(
                                ), emb_gradient))

    def test_simplenet_shared_embedding_clear_gradient(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    for is_sparse in [True, False]:
                        with fluid.dygraph.guard(place):
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array(
                                [[[1], [2]], [[2], [1]]]).astype('int64')
                            input = base.to_variable(input_word)
                            simplenet = SimpleNetSharedEmbedding(
                                "SimpleNetSharedEmbedding", 20, 32, dtype,
                                is_sparse)
                            input_emb, emb = simplenet(input)
                            input_emb.backward(backward_strategy)
                            adam.minimize(input_emb)
                            emb_gradient = emb._w.gradient()
                            simplenet.clear_gradients()

                            input_emb, emb = simplenet(input)
                            input_emb.backward(backward_strategy)

                            if is_sparse:
                                self.assertTrue(
                                    np.array_equal(emb._w.gradient()[0],
                                                   emb_gradient[0]))
                            else:
                                self.assertTrue(
                                    np.array_equal(emb._w.gradient(),
                                                   emb_gradient))

    def test_simplenet_only_embedding_clear_gradient(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            for dtype in ["float32", "float64"]:
                for sort_sum_gradient in [True, False]:
                    for is_sparse in [True, False]:
                        with fluid.dygraph.guard(place):
                            backward_strategy = fluid.dygraph.BackwardStrategy()
                            backward_strategy.sort_sum_gradient = sort_sum_gradient
                            adam = fluid.optimizer.SGDOptimizer(
                                learning_rate=0.001)

                            input_word = np.array(
                                [[[1], [2]], [[2], [1]]]).astype('int64')
                            input = base.to_variable(input_word)
                            simplenet = SimpleNetOnlyEmbedding(
                                "SimpleNetOnlyEmbedding", 20, 32, dtype,
                                is_sparse)
                            input_emb, emb = simplenet(input)
                            input_emb.backward(backward_strategy)
                            emb_gradient = emb._w.gradient()
                            simplenet.clear_gradients()

                            input_emb, emb = simplenet(input)
                            input_emb.backward(backward_strategy)

                            if is_sparse:
                                self.assertTrue(
                                    np.array_equal(emb._w.gradient()[0],
                                                   emb_gradient[0]))
                            else:
                                self.assertTrue(
                                    np.array_equal(emb._w.gradient(),
                                                   emb_gradient))


if __name__ == '__main__':
    unittest.main()
