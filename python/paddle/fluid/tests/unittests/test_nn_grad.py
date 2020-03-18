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


class TestMulGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        prog = fluid.Program()
        with fluid.program_guard(prog):
            x = layers.create_parameter(dtype="float64", shape=[2, 8], name='x')
            y = layers.create_parameter(dtype="float64", shape=[8, 4], name='y')
            z = layers.mul(x=x, y=y)
            gradient_checker.grad_check([x, y], z, place=place)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestSliceOpDoubleGradCheck(unittest.TestCase):
    def func(self, place):
        self.config()

        out = fluid.layers.slice(
            self.inputs, axes=self.axes, starts=self.starts, ends=self.ends)
        gradient_checker.double_grad_check(
            [self.inputs], out, x_init=self.x_arr, place=place)

    def config(self):
        self.starts = [1, 0, -1]
        self.ends = [3, 3, 6]
        self.axes = [0, 1, 2]
        self.x_arr = np.random.random([3, 4, 5, 2]).astype("float64")
        self.inputs = layers.create_parameter(
            dtype="float64", shape=[3, 4, 5, 2], name='x')

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.func(place)


class TestSliceOpDoubleGradCheckCase3(TestSliceOpDoubleGradCheck):
    def config(self):
        self.starts = [1, -1, 1]
        self.ends = [3, 3, 3]
        self.axes = [0, 1, 2]
        self.x_arr = np.random.random([3, 3, 3]).astype("float64")
        self.inputs = layers.create_parameter(
            dtype="float64", shape=[3, 3, 3], name='x3')


class TestReduceMeanWithDimDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [7, 11]
        eps = 0.05
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True
        y = layers.reduce_mean(x, dim=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestMulDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        x_shape = [7, 11]
        y_shape = [11, 9]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', x_shape, False, dtype)
        x.persistable = True
        y = layers.data('y', y_shape, False, dtype)
        y.persistable = True
        out = layers.mul(x, y)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, y_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestReshapeDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        x_shape = [3, 12]
        new_shape = [4, 9]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', x_shape, False, dtype)
        x.persistable = True
        out = layers.reshape(x, new_shape)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], out, x_init=x_arr, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
