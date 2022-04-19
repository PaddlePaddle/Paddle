# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.static as static
import numpy as np
import unittest


def cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean'):
    batch_size, hidden_size = input1.shape
    scores = np.zeros(batch_size)
    for i in range(batch_size):
        z = np.matmul(input1[i], input2[i])
        denom = np.linalg.norm(
            input1[i], ord=2) * np.linalg.norm(
                input2[i], ord=2)
        score = z / denom
        if label[i] == 1:
            scores[i] = 1 - score
        else:
            scores[i] = max(0, score - margin)
    if reduction == 'none':
        return scores
    if reduction == 'mean':
        return np.mean(scores)
    elif reduction == 'sum':
        return np.sum(scores)


class TestFunctionCosineEmbeddingLoss(unittest.TestCase):
    def setUp(self):
        self.input1_np = np.random.random(size=(5, 3)).astype(np.float64)
        self.input2_np = np.random.random(size=(5, 3)).astype(np.float64)
        self.label_np = np.random.randint(
            low=0, high=2, size=5).astype(np.int32)

    def run_dynamic(self):
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)
        dy_result = paddle.nn.functional.cosine_embedding_loss(
            input1, input2, label, margin=0.5, reduction='mean')
        expected1 = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='mean')
        self.assertTrue(np.allclose(dy_result.numpy(), expected1))
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.cosine_embedding_loss(
            input1, input2, label, margin=0.5, reduction='sum')
        expected2 = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='sum')

        self.assertTrue(np.allclose(dy_result.numpy(), expected2))
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.cosine_embedding_loss(
            input1, input2, label, margin=0.5, reduction='none')
        expected3 = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='none')

        self.assertTrue(np.allclose(dy_result.numpy(), expected3))
        self.assertTrue(dy_result.shape, [5])

    def run_static(self, use_gpu=False):
        input1 = static.data(name='input1', shape=[5, 3], dtype='float64')
        input2 = static.data(name='input2', shape=[5, 3], dtype='float64')
        label = static.data(name='label', shape=[5], dtype='int32')
        result0 = paddle.nn.functional.cosine_embedding_loss(
            input1, input2, label, margin=0.5, reduction='none')
        result1 = paddle.nn.functional.cosine_embedding_loss(
            input1, input2, label, margin=0.5, reduction='sum')
        result2 = paddle.nn.functional.cosine_embedding_loss(
            input1, input2, label, margin=0.5, reduction='mean')

        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        exe = static.Executor(place)
        exe.run(static.default_startup_program())
        static_result = exe.run(feed={
            "input1": self.input1_np,
            "input2": self.input2_np,
            "label": self.label_np
        },
                                fetch_list=[result0, result1, result2])
        expected = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='none')

        self.assertTrue(np.allclose(static_result[0], expected))
        expected = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='sum')

        self.assertTrue(np.allclose(static_result[1], expected))
        expected = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='mean')

        self.assertTrue(np.allclose(static_result[2], expected))

    def test_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_dynamic()
        paddle.enable_static()

        with static.program_guard(static.Program()):
            self.run_static()

    def test_gpu(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_dynamic()
        paddle.enable_static()

        with static.program_guard(static.Program()):
            self.run_static(use_gpu=True)

    def test_legal_type(self):
        paddle.disable_static()
        input1 = paddle.to_tensor(self.input1_np.astype(np.float32))
        input2 = paddle.to_tensor(self.input2_np.astype(np.float32))
        label = paddle.to_tensor(self.label_np.astype(np.float32))
        result = paddle.nn.functional.cosine_embedding_loss(
            input1, input2, label, margin=0.5, reduction='mean')
        expected = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='mean')
        self.assertTrue(np.allclose(result.numpy(), expected))

        label = paddle.to_tensor(self.label_np.astype(np.float64))
        result = paddle.nn.functional.cosine_embedding_loss(
            input1, input2, label, margin=0.5, reduction='mean')
        expected = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='mean')
        self.assertTrue(np.allclose(result.numpy(), expected))

    def test_errors(self):
        paddle.disable_static()
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)

        def test_label_shape_error():
            label = paddle.to_tensor(
                np.random.randint(
                    low=0, high=2, size=(2, 3)))
            paddle.nn.functional.cosine_embedding_loss(
                input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_label_shape_error)

        def test_input_shape1D_error():
            input1 = paddle.to_tensor(self.input1_np[0])
            label = paddle.to_tensor(np.ndarray([1]))
            paddle.nn.functional.cosine_embedding_loss(
                input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_input_shape1D_error)

        def test_input_shape2D_error():
            input1 = paddle.to_tensor(
                np.random.random(size=(2, 3, 4)).astype(np.float64))
            input2 = paddle.to_tensor(
                np.random.random(size=(2, 3, 4)).astype(np.float64))
            paddle.nn.functional.cosine_embedding_loss(
                input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_input_shape2D_error)

        def test_label_value_error():
            label = paddle.to_tensor(np.ndarray([-1, -2]))
            paddle.nn.functional.cosine_embedding_loss(
                input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_label_value_error)

        def test_input_type_error():
            input1 = paddle.to_tensor(self.input1_np.astype(np.int64))
            paddle.nn.functional.cosine_embedding_loss(
                input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_input_type_error)

        def test_label_type_error():
            label = paddle.to_tensor(self.label_np.astype(np.int16))
            paddle.nn.functional.cosine_embedding_loss(
                input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_label_type_error)


class TestClassCosineEmbeddingLoss(unittest.TestCase):
    def setUp(self):
        self.input1_np = np.random.random(size=(5, 3)).astype(np.float64)
        self.input2_np = np.random.random(size=(5, 3)).astype(np.float64)
        self.label_np = np.random.randint(
            low=0, high=2, size=5).astype(np.int32)

    def run_dynamic(self):
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)
        CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(
            margin=0.5, reduction='mean')
        dy_result = CosineEmbeddingLoss(input1, input2, label)
        expected1 = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='mean')

        self.assertTrue(np.allclose(dy_result.numpy(), expected1))
        self.assertTrue(dy_result.shape, [1])

    def run_static(self):
        input1 = static.data(name='input1', shape=[5, 3], dtype='float64')
        input2 = static.data(name='input2', shape=[5, 3], dtype='float64')
        label = static.data(name='label', shape=[5], dtype='int32')
        CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(
            margin=0.5, reduction='mean')
        result = CosineEmbeddingLoss(input1, input2, label)

        place = paddle.CPUPlace()
        exe = static.Executor(place)
        exe.run(static.default_startup_program())
        static_result = exe.run(feed={
            "input1": self.input1_np,
            "input2": self.input2_np,
            "label": self.label_np
        },
                                fetch_list=[result])
        expected = cosine_embedding_loss(
            self.input1_np,
            self.input2_np,
            self.label_np,
            margin=0.5,
            reduction='mean')

        self.assertTrue(np.allclose(static_result[0], expected))

    def test_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_dynamic()
        paddle.enable_static()

        with static.program_guard(static.Program()):
            self.run_static()

    def test_errors(self):
        def test_margin_error():
            CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(
                margin=2, reduction='mean')

        self.assertRaises(ValueError, test_margin_error)

        def test_reduction_error():
            CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(
                margin=2, reduction='reduce_mean')

        self.assertRaises(ValueError, test_reduction_error)


if __name__ == "__main__":
    unittest.main()
