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


class TestConvDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 4, 7, 8]
        eps = 0.005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype)
        y = layers.conv2d(x, 4, 1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = fluid.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check(
            [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestConvDoubleGradCheckTest1(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype)
        y = layers.conv2d(x, 4, 1, padding=1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = fluid.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check(
            [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestConv3DDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 4, 3, 4, 2]
        eps = 0.005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype)
        y = layers.conv3d(x, 4, 1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = fluid.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check(
            [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestConv3DDoubleGradCheckTest1(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 4, 5, 3, 2]
        eps = 0.005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype)
        y = layers.conv3d(x, 4, 1, padding=1, bias_attr=False)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = fluid.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        gradient_checker.double_grad_check(
            [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
