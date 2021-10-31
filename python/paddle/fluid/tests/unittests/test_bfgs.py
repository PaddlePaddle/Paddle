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

import paddle
from paddle.optimizer.functional import bfgs_iterates, bfgs_optimize

class TestBFGS(unittest.TestCase):
    
    def test_quadratic(self):
        
        input_shape = [10]
        minimum = paddle.rand(input_shape, dtype='float')
        scales = paddle.exp(paddle.rand(input_shape))

        def f(x):
            return paddle.sum(scales * paddle.square(x - minimum))

        x0 = paddle.ones_like(minimum, dtype='float')

        result = bfgs_optimize(f, x0, dtype='float')

        print(result)

        self.assertTrue(result.converged)


import paddle
from paddle.optimizer.functional import bfgs_iterates, bfgs_optimize

input_shape = [10]
minimum = paddle.rand(input_shape, dtype='float')
scales = paddle.exp(paddle.rand(input_shape))

def f(x):
    return paddle.sum(scales * paddle.square(x - minimum))

x0 = paddle.ones_like(minimum, dtype='float')

result = bfgs_optimize(f, x0, dtype='float')