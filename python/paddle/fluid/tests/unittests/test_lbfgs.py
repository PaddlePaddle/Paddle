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

import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid

from paddle.optimizer.functional.line_search import strong_wolfe
from paddle.optimizer.functional.bfgs import miminize_bfgs

np.random.seed(123)


class TestLbfgs(unittest.TestCase):
    def test_static_graph(self,
                          func,
                          x0,
                          H0=None,
                          line_search_fn='strong_wolfe'):
        dimension = x0.shape[0]
        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            X = paddle.static.data(name='x', shape=[dimension], dtype='float32')
            Y = miminize_lbfgs(
                func,
                X,
                initial_inverse_hessian_estimate=H0,
                line_search_fn=line_search_fn)

        exe = fluid.Executor()
        exe.run(startup)
        return exe.run(main, feed={'x': x0}, fetch_list=[Y])

    def test_dynamic_graph(self,
                           func,
                           x0,
                           H0=None,
                           line_search_fn='strong_wolfe'):
        paddle.disable_static()
        return miminize_lbfgs(
            func,
            x0,
            initial_inverse_hessian_estimate=H0,
            line_search_fn=line_search_fn)

    def test_quadratic_nd(self):
        for dimension in [1, 2, 20]:
            minimum = np.random.random(size=[dimension]).astype('float32')
            scale = np.exp(np.random.random(size=[dimension]).astype('float32'))

            def func2(x):
                minimum_ = paddle.assign(minimum)
                scale_ = paddle.assign(scale)
                return paddle.sum(
                    paddle.multiply(scale_, (F.square_error_cost(x, minimum_))))

            x0 = np.random.random(size=[dimension]).astype('float32')
            results = self.test_static_graph(func2, x0)
            self.assertTrue(np.allclose(minimum, results[2]))
            self.assertTrue(results[0][0])

            results = self.test_dynamic_graph(func2, x0)
            self.assertTrue(np.allclose(minimum, results[2].numpy()))
            self.assertTrue(results[0])

    def test_inf_minima(self):
        extream_point = np.array([-1, 2]).astype('float32')

        def func(x):
            # df = 3(x - 1.01)(x - 0.99)
            # f = x^3 - 3x^2 + 3*1.01*0.99x
            return x * x * x / 3.0 - (
                extream_point[0] + extream_point[1]
            ) * x * x / 2 + extream_point[0] * extream_point[1] * x

        x0 = np.array([-1.7]).astype('float32')
        results = self.test_static_graph(func, x0)
        self.assertFalse(results[0][0])

        results = self.test_dynamic_graph(func, x0)
        self.assertFalse(results[0])

    def test_multi_minima(self):
        def func(x):
            # df = 12(x + 1.1)(x - 0.2)(x - 0.8)
            # f = 3*x^4+0.4*x^3-5.46*x^2+2.112*x
            # minimum = -1.1 or 0.8. 
            # All these minima may be reached from appropriate starting points.
            return 3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x

        x0 = np.array([0.82], dtype='float32')
        x1 = np.array([-1.3], dtype='float32')

        results = self.test_static_graph(func, x0)
        self.assertTrue(np.allclose(0.8, results[2]))

        results = self.test_static_graph(func, x1)
        self.assertTrue(np.allclose(-1.1, results[2]))

    def test_rosenbrock(self):
        """Tests BFGS on the Rosenbrock function.
        The Rosenbrock function is a standard optimization test case. In two
        dimensions, the function is (a, b > 0):
        f(x, y) = (a - x)^2 + b (y - x^2)^2
        The function has a global minimum at (a, a^2). This minimum lies inside
        a parabolic valley (y = x^2).
        """
        a = np.random.random(size=[1]).astype('float32')
        minimum = [a.item(), (a**2).item()]
        b = np.random.random(size=[1]).astype('float32')

        def func(position):
            x, y = position[0], position[1]
            c = (a - x)**2 + b * (y - x**2)**2
            # the return cant be np array[1], or in jacobin will cause flat error
            return c[0]

        x0 = np.random.random(size=[2]).astype('float32')

        results = self.test_static_graph(func, x0)
        self.assertTrue(np.allclose(minimum, results[2]))

        results = self.test_dynamic_graph(func, x0)
        self.assertTrue(np.allclose(minimum, results[2]))

    def test_initial_inverse_hessian_estimate(self):
        def func(x):
            return paddle.dot(x, x)

        x0 = np.random.random(size=[2]).astype('float32')
        H0 = np.array([[1.0, 0.0], [0.0, 1.0]]).astype('float32')
        H1 = np.array([[1.0, 2.0], [2.0, 1.0]]).astype('float32')
        H2 = np.array([[1.0, 2.0], [2.0, 1.0]]).astype('float32')

        results = self.test_static_graph(func, x0, H0)
        self.assertTrue(np.allclose([0., 0.], results[2]))
        self.assertTrue(results[0][0])

        self.assertRaises(ValueError, self.test_dynamic_graph, func, x0, H0=H1)
        self.assertRaises(ValueError, self.test_dynamic_graph, func, x0, H0=H2)

    def test_static_line_search_fn(self):
        def func(x):
            return paddle.dot(x, x)

        x0 = np.random.random(size=[2]).astype('float32')

        self.assertRaises(
            NotImplementedError,
            self.test_static_graph,
            func,
            x0,
            line_search_fn='other')
        self.assertRaises(
            NotImplementedError,
            self.test_dynamic_graph,
            func,
            x0,
            line_search_fn='other')


if __name__ == '__main__':
    unittest.main()
