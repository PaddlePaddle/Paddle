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
import paddle.fluid as fluid
import numpy as np
import unittest


def cosine_embedding_loss(input1: np.ndarray, input2: np.ndarray, label: np.ndarray, margin=0.5, reduction='mean'):
    batch_size, hidden_size = input1.shape
    scores = np.zeros(batch_size)
    for i in range(batch_size):
        z = np.matmul(input1[i], input2[i])
        denom = np.linalg.norm(input1[i], ord=2) * np.linalg.norm(input2[i], ord=2)
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
        self.input1_np = np.random.random(size=(2, 3)).astype(np.float32)
        self.input2_np = np.random.random(size=(2, 3)).astype(np.float32)
        self.label_np = np.random.randint(low=0, high=2, size=2)

    def run_dynamic(self):
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)
        dy_result = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        expected1 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='mean')
        self.assertTrue(np.allclose(dy_result.numpy(), expected1))
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='sum')
        expected2 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='sum')
        self.assertTrue(np.allclose(dy_result.numpy(), expected2))
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='none')
        expected3 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='none')
        self.assertTrue(np.allclose(dy_result.numpy(), expected3))
        self.assertTrue(dy_result.shape, [2])

    def run_static(self, use_gpu=False):
        input1 = paddle.fluid.data(name='input1', shape=[2, 3], dtype='float32')
        input2 = paddle.fluid.data(name='input2', shape=[2, 3], dtype='float32')
        label = paddle.fluid.data(name='label', shape=[2], dtype='int32')
        result0 = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='none')
        result1 = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='sum')
        result2 = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        static_result = exe.run(
            feed={"input1": self.input1_np,
                  "input2": self.input2_np,
                  "label": self.label_np},
            fetch_list=[result0, result1, result2])
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='none')
        self.assertTrue(np.allclose(static_result[0], expected))
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='sum')
        self.assertTrue(np.allclose(static_result[1], expected))
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='mean')
        self.assertTrue(np.allclose(static_result[2], expected))

    def test_cpu(self):
        paddle.disable_static(place=paddle.fluid.CPUPlace())
        self.run_dynamic()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.fluid.CUDAPlace(0))
        self.run_dynamic()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static(use_gpu=True)

    def test_errors(self):
        paddle.disable_static()
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)

        def test_label_shape_error():
            label = paddle.to_tensor(np.random.randint(low=0, high=1, size=(2, 3)))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_label_shape_error)

        def test_input_shape1D_error():
            input1 = paddle.to_tensor(self.input1_np[0])
            label = paddle.to_tensor(np.ndarray([1]))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_input_shape1D_error)

        def test_input_shape2D_error():
            input1 = paddle.to_tensor(np.random.random(size=(2, 3, 4)).astype(np.float32))
            input2 = paddle.to_tensor(np.random.random(size=(2, 3, 4)).astype(np.float32))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_input_shape2D_error)

        def test_label_value_error():
            label = paddle.to_tensor(np.ndarray([-1, -2]))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')

        self.assertRaises(ValueError, test_label_value_error)


class TestClassCosineEmbeddingLoss(unittest.TestCase):
    def setUp(self):
        self.input1_np = np.random.random(size=(2, 3)).astype(np.float32)
        self.input2_np = np.random.random(size=(2, 3)).astype(np.float32)
        self.label_np = np.random.randint(low=0, high=2, size=2)

    def run_dynamic(self):
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)
        CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(margin=0.5, reduction='mean')
        dy_result = CosineEmbeddingLoss(input1, input2, label)
        expected1 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='mean')
        self.assertTrue(np.allclose(dy_result.numpy(), expected1))
        self.assertTrue(dy_result.shape, [1])

        CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(margin=0.5, reduction='sum')
        dy_result = CosineEmbeddingLoss(input1, input2, label)
        expected2 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='sum')
        self.assertTrue(np.allclose(dy_result.numpy(), expected2))
        self.assertTrue(dy_result.shape, [1])

        CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(margin=0.5, reduction='none')
        dy_result = CosineEmbeddingLoss(input1, input2, label)
        expected3 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='none')
        self.assertTrue(np.allclose(dy_result.numpy(), expected3))
        self.assertTrue(dy_result.shape, [1, 2])

    def run_static(self, use_gpu=False):
        input1 = paddle.fluid.data(name='input1', shape=[2, 3], dtype='float32')
        input2 = paddle.fluid.data(name='input2', shape=[2, 3], dtype='float32')
        label = paddle.fluid.data(name='label', shape=[2], dtype='int32')
        CosineEmbeddingLoss0 = paddle.nn.loss.CosineEmbeddingLoss(margin=0.5, reduction='none')
        CosineEmbeddingLoss1 = paddle.nn.loss.CosineEmbeddingLoss(margin=0.5, reduction='sum')
        CosineEmbeddingLoss2 = paddle.nn.loss.CosineEmbeddingLoss(margin=0.5, reduction='mean')
        result0 = CosineEmbeddingLoss0(input1, input2, label)
        result1 = CosineEmbeddingLoss1(input1, input2, label)
        result2 = CosineEmbeddingLoss2(input1, input2, label)

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        static_result = exe.run(
            feed={"input1": self.input1_np,
                  "input2": self.input2_np,
                  "label": self.label_np},
            fetch_list=[result0, result1, result2])
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='none')
        self.assertTrue(np.allclose(static_result[0], expected))
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='sum')
        self.assertTrue(np.allclose(static_result[1], expected))
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='mean')
        self.assertTrue(np.allclose(static_result[2], expected))

    def test_cpu(self):
        paddle.disable_static(place=paddle.fluid.CPUPlace())
        self.run_dynamic()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.fluid.CUDAPlace(0))
        self.run_dynamic()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static(use_gpu=True)

    def test_errors(self):
        def test_margin_error():
            CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(margin=2, reduction='mean')

        self.assertRaises(ValueError, test_margin_error)

        def test_reduction_error():
            CosineEmbeddingLoss = paddle.nn.loss.CosineEmbeddingLoss(margin=2, reduction='reduce_mean')

        self.assertRaises(ValueError, test_reduction_error)


if __name__ == "__main__":
    unittest.main()
