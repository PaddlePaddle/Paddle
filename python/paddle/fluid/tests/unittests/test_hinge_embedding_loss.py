# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import unittest

np.random.seed(42)


class TestFunctionalHingeEmbeddingLoss(unittest.TestCase):
    def setUp(self):
        self.delta = 1.0
        self.shape = (10, 10, 5)
        self.input_np = np.random.random(size=self.shape).astype(np.float32)
        # get label elem in {1., -1.}
        self.label_np = 2 * np.random.randint(0, 2, size=self.shape) - 1.
        # get wrong label elem not in {1., -1.}
        self.wrong_label = paddle.randint(-3, 3, shape=self.shape)

    def run_dynamic_check(self):
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np, dtype=paddle.float32)
        dy_result = paddle.nn.functional.hinge_embedding_loss(input, label)
        expected = np.mean(
            np.where(label.numpy() == 1.,
                     input.numpy(), np.maximum(0., self.delta - input.numpy())))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.hinge_embedding_loss(
            input, label, reduction='sum')
        expected = np.sum(
            np.where(label.numpy() == 1.,
                     input.numpy(), np.maximum(0., self.delta - input.numpy())))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.hinge_embedding_loss(
            input, label, reduction='none')
        expected = np.where(label.numpy() == 1.,
                            input.numpy(),
                            np.maximum(0., self.delta - input.numpy()))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, self.shape)

    def test_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_dynamic_check()

    def test_gpu(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_dynamic_check()

    # test case the raise message
    def test_reduce_errors(self):
        def test_value_error():
            loss = paddle.nn.functional.hinge_embedding_loss(
                self.input_np, self.label_np, reduction='reduce_mean')

        self.assertRaises(ValueError, test_value_error)

    def test_label_errors(self):
        paddle.disable_static()

        def test_value_error():
            loss = paddle.nn.functional.hinge_embedding_loss(
                paddle.to_tensor(self.input_np), self.wrong_label)

        self.assertRaises(ValueError, test_value_error)


class TestClassHingeEmbeddingLoss(unittest.TestCase):
    def setUp(self):
        self.delta = 1.0
        self.shape = (10, 10, 5)
        self.input_np = np.random.random(size=self.shape).astype(np.float32)
        # get label elem in {1., -1.}
        self.label_np = 2 * np.random.randint(0, 2, size=self.shape) - 1.
        # get wrong label elem not in {1., -1.}
        self.wrong_label = paddle.randint(-3, 3, shape=self.shape)

    def run_dynamic_check(self):
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np, dtype=paddle.float32)
        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss()
        dy_result = hinge_embedding_loss(input, label)
        expected = np.mean(
            np.where(label.numpy() == 1.,
                     input.numpy(), np.maximum(0., self.delta - input.numpy())))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [1])

        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(
            reduction='sum')
        dy_result = hinge_embedding_loss(input, label)
        expected = np.sum(
            np.where(label.numpy() == 1.,
                     input.numpy(), np.maximum(0., self.delta - input.numpy())))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, [1])

        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(
            reduction='none')
        dy_result = hinge_embedding_loss(input, label)
        expected = np.where(label.numpy() == 1.,
                            input.numpy(),
                            np.maximum(0., self.delta - input.numpy()))
        self.assertTrue(np.allclose(dy_result.numpy(), expected))
        self.assertTrue(dy_result.shape, self.shape)

    def test_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_dynamic_check()

    def test_gpu(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_dynamic_check()

    # test case the raise message
    def test_reduce_errors(self):
        def test_value_error():
            hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(
                reduction='reduce_mean')
            loss = hinge_embedding_loss(self.input_np, self.label_np)

        self.assertRaises(ValueError, test_value_error)

    def test_label_errors(self):
        paddle.disable_static()

        def test_value_error():
            hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss()
            loss = hinge_embedding_loss(
                paddle.to_tensor(self.input_np), self.wrong_label)

        self.assertRaises(ValueError, test_value_error)


if __name__ == "__main__":
    unittest.main()
