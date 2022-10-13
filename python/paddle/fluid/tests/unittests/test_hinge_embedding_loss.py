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

import paddle
import numpy as np
import unittest
from paddle.static import Program, program_guard

np.random.seed(42)


def calc_hinge_embedding_loss(input, label, margin=1.0, reduction='mean'):
    result = np.where(label == -1., np.maximum(0., margin - input), 0.) + \
             np.where(label == 1., input, 0.)
    if reduction == 'none':
        return result
    elif reduction == 'sum':
        return np.sum(result)
    elif reduction == 'mean':
        return np.mean(result)


class TestFunctionalHingeEmbeddingLoss(unittest.TestCase):

    def setUp(self):
        self.margin = 1.0
        self.shape = (10, 10, 5)
        self.input_np = np.random.random(size=self.shape).astype(np.float64)
        # get label elem in {1., -1.}
        self.label_np = 2 * np.random.randint(0, 2, size=self.shape) - 1.

    def run_dynamic_check(self, place=paddle.CPUPlace()):
        paddle.disable_static(place=place)
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np, dtype=paddle.float64)

        dy_result = paddle.nn.functional.hinge_embedding_loss(input, label)
        expected = calc_hinge_embedding_loss(self.input_np, self.label_np)
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.hinge_embedding_loss(input,
                                                              label,
                                                              reduction='sum')
        expected = calc_hinge_embedding_loss(self.input_np,
                                             self.label_np,
                                             reduction='sum')
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertTrue(dy_result.shape, [1])

        dy_result = paddle.nn.functional.hinge_embedding_loss(input,
                                                              label,
                                                              reduction='none')
        expected = calc_hinge_embedding_loss(self.input_np,
                                             self.label_np,
                                             reduction='none')
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertTrue(dy_result.shape, self.shape)

    def run_static_check(self, place=paddle.CPUPlace):
        paddle.enable_static()
        for reduction in ['none', 'mean', 'sum']:
            expected = calc_hinge_embedding_loss(self.input_np,
                                                 self.label_np,
                                                 reduction=reduction)
            with program_guard(Program(), Program()):
                input = paddle.static.data(name="input",
                                           shape=self.shape,
                                           dtype=paddle.float64)
                label = paddle.static.data(name="label",
                                           shape=self.shape,
                                           dtype=paddle.float64)
                st_result = paddle.nn.functional.hinge_embedding_loss(
                    input, label, reduction=reduction)
                exe = paddle.static.Executor(place)
                result_numpy, = exe.run(feed={
                    "input": self.input_np,
                    "label": self.label_np
                },
                                        fetch_list=[st_result])
                np.testing.assert_allclose(result_numpy, expected, rtol=1e-05)

    def test_cpu(self):
        self.run_dynamic_check(place=paddle.CPUPlace())
        self.run_static_check(place=paddle.CPUPlace())

    def test_gpu(self):
        if not paddle.is_compiled_with_cuda():
            return
        self.run_dynamic_check(place=paddle.CUDAPlace(0))
        self.run_static_check(place=paddle.CUDAPlace(0))

    # test case the raise message
    def test_reduce_errors(self):

        def test_value_error():
            loss = paddle.nn.functional.hinge_embedding_loss(
                self.input_np, self.label_np, reduction='reduce_mean')

        self.assertRaises(ValueError, test_value_error)


class TestClassHingeEmbeddingLoss(unittest.TestCase):

    def setUp(self):
        self.margin = 1.0
        self.shape = (10, 10, 5)
        self.input_np = np.random.random(size=self.shape).astype(np.float64)
        # get label elem in {1., -1.}
        self.label_np = 2 * np.random.randint(0, 2, size=self.shape) - 1.

    def run_dynamic_check(self, place=paddle.CPUPlace()):
        paddle.disable_static(place=place)
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np, dtype=paddle.float64)
        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss()
        dy_result = hinge_embedding_loss(input, label)
        expected = calc_hinge_embedding_loss(self.input_np, self.label_np)
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertTrue(dy_result.shape, [1])

        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(
            reduction='sum')
        dy_result = hinge_embedding_loss(input, label)
        expected = calc_hinge_embedding_loss(self.input_np,
                                             self.label_np,
                                             reduction='sum')
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertTrue(dy_result.shape, [1])

        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(
            reduction='none')
        dy_result = hinge_embedding_loss(input, label)
        expected = calc_hinge_embedding_loss(self.input_np,
                                             self.label_np,
                                             reduction='none')
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertTrue(dy_result.shape, self.shape)

    def run_static_check(self, place=paddle.CPUPlace):
        paddle.enable_static()
        for reduction in ['none', 'mean', 'sum']:
            expected = calc_hinge_embedding_loss(self.input_np,
                                                 self.label_np,
                                                 reduction=reduction)
            with program_guard(Program(), Program()):
                input = paddle.static.data(name="input",
                                           shape=self.shape,
                                           dtype=paddle.float64)
                label = paddle.static.data(name="label",
                                           shape=self.shape,
                                           dtype=paddle.float64)
                hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(
                    reduction=reduction)
                st_result = hinge_embedding_loss(input, label)
                exe = paddle.static.Executor(place)
                result_numpy, = exe.run(feed={
                    "input": self.input_np,
                    "label": self.label_np
                },
                                        fetch_list=[st_result])
                np.testing.assert_allclose(result_numpy, expected, rtol=1e-05)

    def test_cpu(self):
        self.run_dynamic_check(place=paddle.CPUPlace())
        self.run_static_check(place=paddle.CPUPlace())

    def test_gpu(self):
        if not paddle.is_compiled_with_cuda():
            return
        self.run_dynamic_check(place=paddle.CUDAPlace(0))
        self.run_static_check(place=paddle.CUDAPlace(0))

    # test case the raise message
    def test_reduce_errors(self):

        def test_value_error():
            hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(
                reduction='reduce_mean')
            loss = hinge_embedding_loss(self.input_np, self.label_np)

        self.assertRaises(ValueError, test_value_error)


if __name__ == "__main__":
    unittest.main()
