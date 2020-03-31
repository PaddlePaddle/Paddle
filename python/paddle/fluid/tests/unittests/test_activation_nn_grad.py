#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import gradient_checker

from decorator_helper import prog_scope


class TestReluDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True
        y = layers.relu(x)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.02

        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestLeakyReluDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.005
        alpha = 0.2
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True

        y = layers.leaky_relu(x, alpha=alpha)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.02

        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places = [fluid.CUDAPlace(0)]
        for p in places:
            self.func(p)


class TestELUDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 1e-6
        alpha = 1.1
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True

        y = layers.elu(x, alpha=alpha)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestSqrtDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.0001
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True

        y = layers.sqrt(x)
        x_arr = np.random.uniform(0.1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places = [fluid.CUDAPlace(0)]
        for p in places:
            self.func(p)


class TestSquareDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 7, 9]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True
        y = layers.square(x)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
