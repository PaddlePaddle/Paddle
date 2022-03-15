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
from bfgs import miminize_bfgs
from utils import _value_and_gradient
import numpy as np
import paddle.nn.functional as F
import paddle.fluid as fluid


class TestBfgs(unittest.TestCase):
    def test_static_quadratic_1d(self):
        minimun = 1.0

        def func(x):
            # df = 2 * (x - minimun)
            return (x - minimun)**2

        position = np.array([789.9], dtype='float32')
        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[1], dtype='float32')
            Y = miminize_bfgs(func, position)

        exe = fluid.Executor()
        exe.run(startup)
        results = exe.run(main, feed={'x': position}, fetch_list=[Y])
        print('position: {}\n g: {}\n H: {}'.format(results[0], results[2],
                                                    results[3]))
        self.assertTrue(np.allclose(minimun, results[0], rtol=1e-08))

    def test_static_quadratic_2d(self):
        def func(x):
            minimun = paddle.assign(np.array([1.0, 2.0], dtype='float32'))
            scale = paddle.assign(np.array([3.0, 4.0], dtype='float32'))
            return paddle.sum(
                paddle.multiply(scale, F.square_error_cost(x, minimun)))

        position = np.array([1234.567, -345.9876], dtype='float32')
        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[2], dtype='float32')
            Y = miminize_bfgs(func, position)

        exe = fluid.Executor()
        exe.run(startup)
        results = exe.run(main, feed={'x': position}, fetch_list=[Y])
        print('num_func_calls: {} \n position: {}\n value: {} \n g: {}\n H: {} \n'.format(results[0], results[1], results[2], results[3],
                                                    results[3]))

        self.assertTrue(np.allclose([1.0, 2.0], results[1], rtol=1e-08))

    def test_static_inf_minima(self):
        extream_point = paddle.to_tensor([-1, 2])

        def func(x):
            # df = 3(x - 1.01)(x - 0.99) = 3x^2 - 3*2x + 3*1.01*0.99
            # f = x^3 - 3x^2 + 3*1.01*0.99x
            return x * x * x / 3.0 - (
                extream_point[0] + extream_point[1]
            ) * x * x / 2 + extream_point[0] * extream_point[1] * x

        position = paddle.to_tensor(3.15)
        results = miminize_bfgs(func, position)
        print('position: {}\n g: {}\n H: {}'.format(results[0], results[2],
                                                    results[3]))

        self.assertAlmostEqual(float("-inf"), position.numpy())

    def test_static_multi_minima_with_tf(self):
        def func(x):
            # df = 12(x + 1.1)(x - 0.2)(x - 0.8)
            # f = 3*x^4+0.4*x^3-5.46*x^2+2.112*x
            return 3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x

        position = np.array([3.6], dtype='float32')
        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[1], dtype='float32')
            Y = miminize_bfgs(func, position)

        exe = fluid.Executor()
        exe.run(startup)
        results = exe.run(main, feed={'x': position}, fetch_list=[Y])
        print('position: {}\n g: {}\n H: {}'.format(results[0], results[2],
                                                    results[3]))

        self.assertTrue(np.allclose(-1.1, results[0], rtol=1e-08))

        import tensorflow as tf
        import tensorflow_probability as tfp
        fdf = lambda x: ValueAndGradient(x=x, f=3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x, df=12 * x**3 + 1.2 * x**2 - 11.28 * x + 2.112)
        #for position in positions:

    #results = tfp.optimizer.linesearch.hager_zhang(fdf)

    #self.assertTrue(np.allclose(minimun.numpy(), results_paddle.numpy(), rtol=1e-08))

    def test_quadratic_1d(self):
        minimun = paddle.to_tensor([1.0])
        scale = paddle.to_tensor([1.0])

        def func(x):
            return paddle.sum(scale.multiply(F.square_error_cost(x, minimun)))

        position = paddle.to_tensor([789.9])
        results = miminize_bfgs(func, position)
        print('position: {}\n g: {}\n H: {}'.format(results[0], results[2],
                                                    results[3]))

        self.assertTrue(
            np.allclose(
                minimun.numpy(), results[0].numpy(), rtol=1e-06))

    def test_quadratic_2d(self):
        minimun = paddle.to_tensor([1.0, 2.0])
        scale = paddle.to_tensor([3.0, 4.0])

        def func(x):
            return paddle.sum(scale.multiply(F.square_error_cost(x, minimun)))

        position = paddle.to_tensor([2.0, 3.0])
        results = miminize_bfgs(func, position)
        print('num_func_calls: {} \n position: {}\n value: {} \n g: {}\n H: {} \n'.format(results[0], results[1], results[2], results[3],
                                                    results[3]))

        self.assertTrue(
            np.allclose(
                minimun.numpy(), results[1].numpy(), rtol=1e-08))

    def test_inf_minima(self):
        extream_point = paddle.to_tensor([-1, 2])

        def func(x):
            # df = 3(x - 1.01)(x - 0.99) = 3x^2 - 3*2x + 3*1.01*0.99
            # f = x^3 - 3x^2 + 3*1.01*0.99x
            return x * x * x / 3.0 - (
                extream_point[0] + extream_point[1]
            ) * x * x / 2 + extream_point[0] * extream_point[1] * x

        position = paddle.to_tensor(3.15)
        results = miminize_bfgs(func, position)
        print('position: {}\n g: {}\n H: {}'.format(results[0], results[2],
                                                    results[3]))

        self.assertAlmostEqual(float("-inf"), position.numpy())

    def test_multi_minima_with_tf(self):
        minimun = paddle.to_tensor([1.0, 2.0])

        def func(x):
            # df = 12(x + 1.1)(x - 0.2)(x - 0.8)
            # f = 3*x^4+0.4*x^3-5.46*x^2+2.112*x
            return 3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x

        position = paddle.to_tensor(0.9)
        results = miminize_bfgs(func, position)
        print('position: {}\n g: {}\n H: {}'.format(results[0], results[2],
                                                    results[3]))

        self.assertTrue(np.allclose(-1.1, results[0].numpy(), rtol=1e-07))

        import tensorflow as tf
        import tensorflow_probability as tfp
        fdf = lambda x: ValueAndGradient(x=x, f=3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x, df=12 * x**3 + 1.2 * x**2 - 11.28 * x + 2.112)
        #for position in positions:

    #results = tfp.optimizer.linesearch.hager_zhang(fdf)

    #self.assertTrue(np.allclose(minimun.numpy(), results_paddle.numpy(), rtol=1e-08))


test = TestBfgs()
test.test_quadratic_2d()
