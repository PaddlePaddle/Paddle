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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.framework import in_pir_mode


class TestFunctionalL1Loss(unittest.TestCase):
    def setUp(self):
        self.input_np = np.random.random(size=(10, 10, 5)).astype(np.float32)
        self.label_np = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self):
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        dy_result = paddle.nn.functional.l1_loss(input, label)
        expected = np.mean(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])

        dy_result = paddle.nn.functional.l1_loss(input, label, reduction='sum')
        expected = np.sum(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])

        dy_result = paddle.nn.functional.l1_loss(input, label, reduction='none')
        expected = np.abs(self.input_np - self.label_np)
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [10, 10, 5])

    def run_static(self, use_gpu=False):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input = paddle.static.data(
                name='input', shape=[10, 10, 5], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[10, 10, 5], dtype='float32'
            )
            result0 = paddle.nn.functional.l1_loss(input, label)
            result1 = paddle.nn.functional.l1_loss(
                input, label, reduction='sum'
            )
            result2 = paddle.nn.functional.l1_loss(
                input, label, reduction='none'
            )
            y = paddle.nn.functional.l1_loss(input, label, name='aaa')

            place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
            exe = paddle.static.Executor(place)
            static_result = exe.run(
                feed={"input": self.input_np, "label": self.label_np},
                fetch_list=[result0, result1, result2],
            )

            expected = np.mean(np.abs(self.input_np - self.label_np))
            np.testing.assert_allclose(static_result[0], expected, rtol=1e-05)
            expected = np.sum(np.abs(self.input_np - self.label_np))
            np.testing.assert_allclose(static_result[1], expected, rtol=1e-05)
            expected = np.abs(self.input_np - self.label_np)
            np.testing.assert_allclose(static_result[2], expected, rtol=1e-05)
            if not in_pir_mode():
                self.assertTrue('aaa' in y.name)

    def test_cpu(self):
        paddle.disable_static(place=paddle.base.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        self.run_static()

    def test_gpu(self):
        if not base.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.base.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        self.run_static(use_gpu=True)

    # test case the raise message
    def test_errors(self):

        def test_value_error():
            input = paddle.static.data(
                name='input', shape=[10, 10, 5], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[10, 10, 5], dtype='float32'
            )
            loss = paddle.nn.functional.l1_loss(
                input, label, reduction='reduce_mean'
            )

        self.assertRaises(ValueError, test_value_error)


class TestClassL1Loss(unittest.TestCase):
    def setUp(self):
        self.input_np = np.random.random(size=(10, 10, 5)).astype(np.float32)
        self.label_np = np.random.random(size=(10, 10, 5)).astype(np.float32)

    def run_imperative(self):
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        l1_loss = paddle.nn.loss.L1Loss()
        dy_result = l1_loss(input, label)
        expected = np.mean(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])

        l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
        dy_result = l1_loss(input, label)
        expected = np.sum(np.abs(self.input_np - self.label_np))
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])

        l1_loss = paddle.nn.loss.L1Loss(reduction='none')
        dy_result = l1_loss(input, label)
        expected = np.abs(self.input_np - self.label_np)
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [10, 10, 5])

    def run_static(self, use_gpu=False):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input = paddle.static.data(
                name='input', shape=[10, 10, 5], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[10, 10, 5], dtype='float32'
            )
            l1_loss = paddle.nn.loss.L1Loss()
            result0 = l1_loss(input, label)
            l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
            result1 = l1_loss(input, label)
            l1_loss = paddle.nn.loss.L1Loss(reduction='none')
            result2 = l1_loss(input, label)
            l1_loss = paddle.nn.loss.L1Loss(name='aaa')
            result3 = l1_loss(input, label)

            place = base.CUDAPlace(0) if use_gpu else base.CPUPlace()
            exe = paddle.static.Executor(place)
            static_result = exe.run(
                feed={"input": self.input_np, "label": self.label_np},
                fetch_list=[result0, result1, result2],
            )

            expected = np.mean(np.abs(self.input_np - self.label_np))
            np.testing.assert_allclose(static_result[0], expected, rtol=1e-05)
            expected = np.sum(np.abs(self.input_np - self.label_np))
            np.testing.assert_allclose(static_result[1], expected, rtol=1e-05)
            expected = np.abs(self.input_np - self.label_np)
            np.testing.assert_allclose(static_result[2], expected, rtol=1e-05)

            if not in_pir_mode():
                self.assertTrue('aaa' in result3.name)

    def test_cpu(self):
        paddle.disable_static(place=paddle.base.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        self.run_static()

    def test_gpu(self):
        if not base.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.base.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        self.run_static(use_gpu=True)

    # test case the raise message
    def test_errors(self):

        def test_value_error():
            loss = paddle.nn.loss.L1Loss(reduction="reduce_mean")

        self.assertRaises(ValueError, test_value_error)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
