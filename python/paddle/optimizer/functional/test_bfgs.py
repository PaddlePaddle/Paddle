# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from line_search import strong_wolfe
from utils import _value_and_gradient
import numpy as np
import paddle.nn.functional as F


class TestLinesearch(unittest.TestCase):
    def test_static(self):
        minimun = 1.2

        def func(x):
            return (x - minimun)**2

        def grad(x):
            return 2 * (x - minimun)

        paddle.enable_static()
        position = 9.1
        for i in range(100):
            results = strong_wolfe(func, position, grad(position))
            print(results)
            position = paddle.add(position, results[0] * grad(position))
            if paddle.abs(x2[0]) + paddle.abs(x2[1]) < 1e-5:
                break

        self.assertAlmostEqual(minimun, results[1])

    def test_quadratic_1d(self):
        minimun = paddle.to_tensor([1.0])
        scale = paddle.to_tensor([1.0])

        def func(x):
            return paddle.sum(scale.multiply(F.square_error_cost(x, minimun)))

        def grad(x):
            return -paddle.multiply(2 * scale, paddle.subtract(x, minimun))

        position = paddle.to_tensor([2.0])
        for i in range(100):
            print(
                '--------------------------------------------------------  iter: {}'.
                format(i))
            print('position: {}'.format(position))
            results = strong_wolfe(func, position, grad(position))
            position = position + results[0] * grad(position)
            print('position: {}, alpha: {}'.format(position, results[0]))
            if paddle.abs(position - minimun) < 1e-6:
                break

        self.assertTrue(
            np.allclose(
                minimun.numpy(), position.numpy(), rtol=1e-06))

    def test_quadratic_2d(self):
        minimun = paddle.to_tensor([1.0, 2.0])
        scale = paddle.to_tensor([3.0, 4.0])

        def func(x):
            return paddle.sum(scale.multiply(F.square_error_cost(x, minimun)))

        def grad(x):
            return -paddle.multiply(2 * scale, paddle.subtract(x, minimun))

        position = paddle.to_tensor([-8.1, -10.2])
        for i in range(100):
            results = strong_wolfe(func, position, grad(position))
            position = position + results[0] * grad(position)
            if paddle.abs(position[0]) + paddle.abs(position[1]) < 1e-5:
                break

        self.assertTrue(
            np.allclose(
                minimun.numpy(), position.numpy(), rtol=1e-08))

    def test_inf_minima(self):
        extream_point = paddle.to_tensor([-1, 2])

        def func(x):
            # df = 3(x - 1.01)(x - 0.99) = 3x^2 - 3*2x + 3*1.01*0.99
            # f = x^3 - 3x^2 + 3*1.01*0.99x
            return x * x * x / 3.0 - (
                extream_point[0] + extream_point[1]
            ) * x * x / 2 + extream_point[0] * extream_point[1] * x

        def grad(x):
            return -(x - extream_point[0]) * (x - extream_point[1])

        position = paddle.to_tensor(3.15)
        for i in range(30):
            print(
                '--------------------------------------------------------  iter: {}'.
                format(i))
            print('position: {}'.format(position))
            results = strong_wolfe(func, position, grad(position))
            position = position + results[0] * grad(position)
            print('{} {} {}'.format(position, results[0], grad(position)))
            if paddle.isinf(position).item():
                break

        self.assertAlmostEqual(float("-inf"), position.numpy())

    def test_multi_minima_with_tf(self):
        minimun = paddle.to_tensor([1.0, 2.0])

        def func(x):
            # df = 12(x + 1.1)(x - 0.2)(x - 0.8)
            # f = 3*x^4+0.4*x^3-5.46*x^2+2.112*x
            return 3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x

        def grad(x):
            return -(12 * x**3 + 1.2 * x**2 - 11.28 * x + 2.112)

        positions = np.arange(-2.0, 2.0, 0.1, dtype='float32')
        results_paddle = []
        for position in positions:
            position = paddle.to_tensor(position)
            for i in range(100):
                #print('--------------------------------------------------------  iter: {}'.format(i))
                #print('position: {}'.format(position))
                results = strong_wolfe(func, position, grad(position))
                position = position + results[0] * grad(position)
                #print('{} {} {}'.format(position, results[0], grad(position)))
                if paddle.abs(results[0] * grad(position)) < 1e-8:
                    results_paddle.append(position.item())
                    break
        print(results_paddle)

        import tensorflow as tf
        import tensorflow_probability as tfp
        fdf = lambda x: ValueAndGradient(x=x, f=3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x, df=12 * x**3 + 1.2 * x**2 - 11.28 * x + 2.112)
        for position in positions:
            results = tfp.optimizer.linesearch.hager_zhang(fdf)

        self.assertTrue(
            np.allclose(
                minimun.numpy(), results_paddle.numpy(), rtol=1e-08))


test = TestLinesearch()
test.test_multi_minima_with_tf()
