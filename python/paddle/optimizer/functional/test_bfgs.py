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

np.random.seed(123)


class TestBfgs(unittest.TestCase):
    def test_static_graph(self, func, x0, H0=None, line_search_fn='strong_wolfe'):
            dimension = x0.shape[0]
            paddle.enable_static()
            main = fluid.Program()
            startup = fluid.Program()
            with fluid.program_guard(main, startup):
                X = paddle.static.data(name='x', shape=[dimension], dtype='float32')
                Y = miminize_bfgs(func, X, initial_inverse_hessian_estimate=H0, line_search_fn=line_search_fn)

            exe = fluid.Executor()
            exe.run(startup)
            return exe.run(main, feed={'x': x0}, fetch_list=[Y])
            

    def test_dynamic_graph(self, func, x0, H0=None, line_search_fn='strong_wolfe'):
        paddle.disable_static()
        return  miminize_bfgs(func, x0, initial_inverse_hessian_estimate=H0, line_search_fn=line_search_fn)

    def test_quadratic_nd(self):
        def func(x):
            return paddle.dot(x + 1.99, x + 3.01)
  
        for dimension in [1, 2, 1000]:
            x0  = np.random.random(size=[dimension]).astype('float32')
            results = self.test_static_graph(func, x0)
            print(results[1])
            self.assertTrue(np.allclose(np.zeros(dimension), results[1]))

            results = self.test_dynamic_graph(func, x0)
            self.assertTrue(np.allclose(np.zeros(dimension), results[1].numpy())) 


    def test_inf_minima(self):
        extream_point = paddle.to_tensor([-1, 2])
        def func(x):
            # df = 3(x - 1.01)(x - 0.99) = 3x^2 - 3*2x + 3*1.01*0.99
            # f = x^3 - 3x^2 + 3*1.01*0.99x
            return x * x * x / 3.0 - (
                extream_point[0] + extream_point[1]
            ) * x * x / 2 + extream_point[0] * extream_point[1] * x

        x0  = np.random.random(size=[2]).astype('float32')

        self.test_static_graph(func, x0)
        self.assertAlmostEqual(float("-inf"), position.numpy())

        self.test_static_graph(func, x0)
        self.assertAlmostEqual(float("-inf"), position.numpy())

        
    def test_multi_minima(self):
        def func(x):
            # df = 12(x + 1.1)(x - 0.2)(x - 0.8)
            # f = 3*x^4+0.4*x^3-5.46*x^2+2.112*x
            # minimum = -1.1 or 0.8
            return 3 * x**4 + 0.4 * x**3 - 5.64 * x**2 + 2.112 * x

        x0 = np.array([0.84], dtype='float32')
        x1 = np.array([-1.3], dtype='float32')

        results = self.test_static_graph(func, x0)
        self.assertTrue(np.allclose(0.8, results[1]))

        results = self.test_static_graph(func, x1)
        self.assertTrue(np.allclose(-1.1, results[1]))

    def test_initial_inverse_hessian_estimate(self):
        def func(x):
            return paddle.dot(x, x)

        x0  = np.random.random(size=[2]).astype('float32')
        H0 = [[1.0, 2.0], [3.0, 1.0]]
        H1 = [[1.0, 2.0], [2.0, 1.0]]
        
        results = self.test_static_graph(func, x0, H0=H0)
        self.assertTrue(np.allclose(0.8, results[1]))
        results = self.test_static_graph(func, x0, H0=H0)
        self.assertTrue(np.allclose(0.8, results[1]))

        
    def test_static_line_search_fn(self):
        def func(x):
            return paddle.dot(x, x)

        x0  = np.random.random(size=[2]).astype('float32')

        self.test_static_graph(func, x0, line_search_fn='other')
        self.assertTrue(np.allclose(0.8, results[1]))
        self.test_static_graph(func, x0, line_search_fn='other')
        self.assertTrue(np.allclose(0.8, results[1]))


test = TestBfgs()
test.test_quadratic_nd()

#if __name__ == '__main__':
    #unittest.main()