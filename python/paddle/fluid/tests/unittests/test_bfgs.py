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

import unittest

import numpy as np
import paddle
from paddle.optimizer.functional import bfgs_iterates, bfgs_optimize
from paddle.optimizer.functional.bfgs import verify_symmetric_positive_definite_matrix

class TestBFGS(unittest.TestCase):

    def setUp(self):
        pass        

    def gen_configs(self):
        dtypes = ['float32', 'float64']
        shapes = {
            '1d1v': [1],
            '1d2v': [2],
            '2d1v': [2, 1],
            '2d2v': [2, 2],
            '1d100v': [100],
            '10d10v': [10, 10]
        }
        for shape, dtype in zip(shapes.values(), dtypes):
            yield shape, dtype

    def test_update_approx_inverse_hessian(self):
        for shape, dtype in self.gen_configs():
            
    
    def _regular_quadratic(self, dtype):
        
        input_shape = [10, 10]
        minimum = paddle.rand(input_shape, dtype=dtype)
        scales = paddle.exp(paddle.rand(input_shape, dtype=dtype))

        def f(x):
            return paddle.sum(scales * paddle.square(x - minimum), axis=-1)

        x0 = paddle.ones_like(minimum, dtype=dtype)

        result = bfgs_optimize(f, x0, dtype=dtype)

        self.assertTrue(paddle.all(result.converged))
        self.assertTrue(paddle.allclose(result.location, minimum))

    def test_quadratic_float32(self):
        paddle.seed(12345)
        self._regular_quadratic('float32')

    def test_quadratic_float64(self):
        paddle.seed(12345)
        self._regular_quadratic('float64')

    def _general_quadratic(self, dtype):
        input_shape = [10, 10]
        minimum = paddle.rand(input_shape, dtype=dtype)
        hessian_shape = input_shape + input_shape[-1:]
        rotation = paddle.rand(hessian_shape, dtype=dtype)
        hessian = paddle.einsum('...ik, ...jk', rotation, rotation)

        verify_symmetric_positive_definite_matrix(hessian)

        def f(x):
            y = paddle.einsum('...i, ...ij, ...j',
                              x - minimum,
                              hessian,
                              x - minimum)

            return y
        
        x0 = paddle.ones_like(minimum, dtype=dtype)

        result = bfgs_optimize(f, x0, dtype=dtype)

        self.assertTrue(paddle.all(result.converged))
        self.assertTrue(paddle.allclose(result.location, minimum))

    def test_general_quadratic_float32(self):
        paddle.seed(12345)
        self._general_quadratic('float32')

    def test_general_quadratic_float64(self):
        paddle.seed(12345)
        self._general_quadratic('float64')

if __name__ == "__main__":
    unittest.main()
