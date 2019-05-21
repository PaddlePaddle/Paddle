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


class TestReluDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 8]
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
        shape = [3, 7]
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


class TestSqrtDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [3, 7]
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


class TestConvDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        shape = [2, 4, 14, 16]
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
        if core.is_compiled_with_cuda():
            places = [fluid.CUDAPlace(0)]
            for p in places:
                self.func(p)


class TestSquareDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [17, 23]
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


class TestElementwiseMulDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 3, 5, 7]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_mul(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


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


class TestElementwiseMulBroadcastDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 3, 5, 7]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_mul(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseAddDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 3, 5, 7]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseAddBroadcastDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 3, 5, 7]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseSubDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 3, 5, 7]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_sub(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseSubBroadcastDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 3, 5, 7]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_sub(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestMulDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
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


class TestElementwiseDivDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 3, 7, 9]
        eps = 0.0001
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_div(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr[np.abs(y_arr) < 0.005] = 0.02

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps, atol=1e-3)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseDivBroadcastDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 3, 7, 9]
        eps = 0.0001
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[1:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_div(x, y, axis=1)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[1:-1]).astype(dtype)
        y_arr[np.abs(y_arr) < 0.005] = 0.02

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps, atol=1e-3)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
