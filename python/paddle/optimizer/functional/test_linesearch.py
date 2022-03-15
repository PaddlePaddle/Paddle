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
import paddle.fluid as fluid
from paddle.fluid.framework import in_dygraph_mode
import collections


class TestLinesearch(unittest.TestCase):
    def test_static2(self):
        minimun = 1.2
        def iner_func(a):
            a[0] += 1
            a[1] += 1
            return a
        def func1(x):
            j = paddle.full(shape=[1], fill_value=0, dtype='int64')
            done = paddle.full(shape=[1], fill_value=0, dtype='int64')
            
            def cond(j, done):
                j_print = paddle.static.Print(j, message="j cond")
                return j < 3

            def body(j, done):
                j = j + 1
                a = [j, j]
                j_print = paddle.static.Print(iner_func([j,j])[1],  message="iner_func()")
                
                    #done_print = paddle.static.Print(done, message="done true_fn")
                #paddle.static.nn.cond(j>0, true_fn, None)
                #done_print = paddle.static.Print(done, message="done body")

                return j, done

            paddle.static.nn.while_loop(cond, body, [j, done])
            j_print = paddle.static.Print(j, message="j out")
            return j

        def func2(x):
            y = 2 * x
            return x, y

        position = [1.0]
        '''
        paddle.disable_static()
        y = func1(paddle.to_tensor(position))
        print(y)'''

        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[-1], dtype='float')
            Y = func1(X)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup)

        feeds = {'x': position}
        Y = exe.run(main, feed=feeds, fetch_list=[Y])
        print(Y)

    def test_static(self):
        minimun = 1.0

        def func(x):
            #return paddle.dot(x,x)
            return (x - minimun)**2

        def grad(x):
            return -2 * (x - minimun)

        position = [2.0]

        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[1], dtype='float32')
            value, pk = _value_and_gradient(func, X)
            Y = strong_wolfe(func, X, -pk)

        exe = fluid.Executor()
        exe.run(startup)
        for i in range(100):
            results = exe.run(main, feed={'x': position}, fetch_list=[Y])
            position = position[0] + results[0] * grad(position[0])
            print('position: {} \n alpha: {}'.format(position[0], results[0]))
            if abs(position - minimun) < 1e-8:
                break

        self.assertTrue(np.allclose(minimun, position, rtol=1e-08))

    def test_static_quadratic_2d(self):
        paddle.enable_static()

        def func(x):
            minimun = paddle.assign(np.array([1.0, 2.0], dtype='float32'))
            scale = paddle.assign(np.array([3.0, 4.0], dtype='float32'))
            return paddle.sum(scale.multiply(F.square_error_cost(x, minimun)))

        def grad(x):
            minimun = paddle.assign(np.array([1.0, 2.0], dtype='float32'))
            scale = paddle.assign(np.array([3.0, 4.0], dtype='float32'))
            return -paddle.multiply(2 * scale, paddle.subtract(x, minimun))

        position = [2.0, 3.0]

        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[2], dtype='float32')
            #value, pk = _value_and_gradient(func, X)
            pk = grad(X)
            Y = strong_wolfe(func, X, pk)

        exe = fluid.Executor()
        exe.run(startup)
        for i in range(100):
            results = exe.run(main,
                              feed={'x': np.array(
                                  position, dtype='float32')},
                              fetch_list=[Y, pk])
            position = position + results[0] * results[-1]
            print('position: {} \n alpha: {} \n pk: {}'.format(
                position, results[0], results[-1]))
            if np.allclose([1.0, 2.0], position, rtol=1e-07):
                break

        self.assertTrue(np.allclose([1.0, 2.0], position, rtol=1e-07))

    def test_static_inf_minima(self):
        extream_point = [-1, 2]

        def func(x):
            # df = 3(x - 1.01)(x - 0.99) = 3x^2 - 3*2x + 3*1.01*0.99
            # f = x^3 - 3x^2 + 3*1.01*0.99x
            return x * x * x / 3.0 - (
                extream_point[0] + extream_point[1]
            ) * x * x / 2 + extream_point[0] * extream_point[1] * x

        def grad(x):
            return -(x - extream_point[0]) * (x - extream_point[1])

        position = 3.6
        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[1], dtype='float32')
            pk = grad(X)
            Y = strong_wolfe(func, X, pk)

        exe = fluid.Executor()
        exe.run(startup)

        for i in range(30):
            print(
                '--------------------------------------------------------  iter: {}'.
                format(i))
            print('position: {}'.format(position))
            results = exe.run(main, feed={'x': position}, fetch_list=[Y, pk])
            position = position + results[0] * results[-1]
            print('position: {} \n alpha: {} \n pk: {}'.format(
                position, results[0], results[-1]))
            if np.isinf(position):
                break

        self.assertAlmostEqual(float("-inf"), position)

    def test_static_multi_minima(self):
        def func(x):
            # df = 12(x + 1.1)(x - 0.2)(x - 0.8)
            # f = 3*x^4+0.4*x^3-5.46*x^2+2.112*x
            #return 3.0 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x
            return 3.0 * paddle.pow(x, 4) + 0.4 * paddle.pow(
                x, 3) - 5.64 * paddle.pow(x, 2) + 2.112 * x

        def grad(x):
            return -(12.0 * x**3 + 1.2 * x**2 - 11.28 * x + 2.112)

        position = -1.9
        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[1], dtype='float32')
            pk = grad(X)
            Y = strong_wolfe(func, X, pk)

        exe = fluid.Executor()
        exe.run(startup)
        for i in range(100):
            print(
                '--------------------------------------------------------  iter: {}'.
                format(i))
            print('position: {}'.format(position))
            results = exe.run(main, feed={'x': position}, fetch_list=[Y, pk])
            position = position + results[0] * results[-1]
            print('position: {} \n alpha: {} \n pk: {}'.format(
                position, results[0], results[-1]))
            if np.allclose(-1.1, position, rtol=1e-04):
                break
        self.assertTrue(np.allclose(-1.1, position, rtol=1e-04))

    def test_quadratic_1d(self):
        minimun = paddle.to_tensor([1.0])
        scale = paddle.to_tensor([1.0])

        def func(x):
            return paddle.sum(scale.multiply(F.square_error_cost(x, minimun)))

        def grad(x):
            return -paddle.multiply(2 * scale, paddle.subtract(x, minimun))

        position = paddle.to_tensor([2.0])
        for i in range(1):
            print(
                '--------------------------------------------------------  iter: {}'.
                format(i))
            print('position: {}'.format(position))
            results = strong_wolfe(func, position, grad(position))
            position = position + results[0] * grad(position)
            print('position: {} \n alpha: {}'.format(position, results[0]))
            if paddle.abs(position - minimun) < 1e-8:
                break

        self.assertTrue(
            np.allclose(
                minimun.numpy(), position.numpy(), rtol=1e-08))

    def test_quadratic_2d(self):
        minimun = paddle.to_tensor([1.0, 2.0])
        scale = paddle.to_tensor([3.0, 4.0])

        def func(x):
            return paddle.sum(scale.multiply(F.square_error_cost(x, minimun)))

        def grad(x):
            return -paddle.multiply(2 * scale, paddle.subtract(x, minimun))

        position = paddle.to_tensor([-8.1, -10.2])
        for i in range(100):
            print(
                '--------------------------------------------------------  iter: {}'.
                format(i))
            print('position: {}'.format(position))
            results = strong_wolfe(func, position, grad(position))
            position = position + results[0] * grad(position)
            print('position: {} \n alpha: {}'.format(position, results[0]))
            if paddle.allclose(minimun, position, rtol=1e-08):
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

        position = paddle.to_tensor(3.6)
        for i in range(30):
            print(
                '--------------------------------------------------------  iter: {}'.
                format(i))
            print('position: {}'.format(position))
            results = strong_wolfe(func, position, grad(position))
            position = position + results[0] * grad(position)
            print('position: {} \n alpha: {}'.format(position, results[0]))
            if paddle.isinf(position).item():
                break

        self.assertAlmostEqual(float("-inf"), position.numpy())

    def test_multi_minima(self):
        def func(x):
            # df = 12(x + 1.1)(x - 0.2)(x - 0.8)
            # f = 3*x^4+0.4*x^3-5.46*x^2+2.112*x
            #return 3.0 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x
            return 3.0 * paddle.pow(x, 4) + 0.4 * paddle.pow(
                x, 3) - 5.64 * paddle.pow(x, 2) + 2.112 * x

        def grad(x):
            return -(12.0 * x**3 + 1.2 * x**2 - 11.28 * x + 2.112)

        #print(func(paddle.to_tensor(-1.09997129,dtype='float64')),func(paddle.to_tensor(-1.09999788,dtype='float64')),func(paddle.to_tensor(-1.1,dtype='float64')))
        position = paddle.to_tensor(-1.09997129, dtype='float32')
        for i in range(100):
            print(
                '--------------------------------------------------------  iter: {}'.
                format(i))
            print('position: {}'.format(position))
            print(func(position))
            results = strong_wolfe(func, position, grad(position))
            position = position + results[0] * grad(position)
            print('position: {} \n alpha: {} \n grad: {}'.format(
                position, results[0], grad(position)))
            if paddle.allclose(
                    paddle.to_tensor(
                        -1.1, dtype='float32'), position, rtol=1e-04):
                break
        self.assertTrue(np.allclose(-1.1, position.numpy(), rtol=1e-04))


test = TestLinesearch()
test.test_static2()
