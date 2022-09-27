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

import unittest
import numpy as np

import paddle.fluid as fluid
import paddle
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import gradient_checker
import paddle.nn.functional as F
from paddle.fluid.framework import _test_eager_guard

from decorator_helper import prog_scope


class TestSigmoidTripleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.0005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype=dtype)
        x.persistable = True
        y = layers.sigmoid(x)
        x_arr = np.random.random(shape).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        gradient_checker.triple_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestSigmoidDoubleGradCheck(unittest.TestCase):

    def sigmoid_wrapper(self, x):
        return fluid.layers.sigmoid(x[0])

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.0005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype=dtype)
        x.persistable = True
        y = layers.sigmoid(x)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.sigmoid_wrapper,
                                                       [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestTanhTripleGradCheck(unittest.TestCase):

    def tanh_wrapper(self, x):
        return paddle.tanh(x[0])

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.0005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype=dtype)
        x.persistable = True
        y = layers.tanh(x)
        x_arr = np.random.random(shape).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        gradient_checker.triple_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(self.tanh_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestTanhDoubleGradCheck(unittest.TestCase):

    def tanh_wrapper(self, x):
        return paddle.tanh(x[0])

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.0005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype=dtype)
        x.persistable = True
        y = paddle.tanh(x)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.tanh_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestAbsDoubleGradCheck(unittest.TestCase):

    def abs_wrapper(self, x):
        return paddle.abs(x[0])

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.0005
        dtype = np.float64
        x = layers.data('x', shape, False, dtype=dtype)
        x.persistable = True
        y = paddle.abs(x)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.abs_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


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

        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestLeakyReluDoubleGradCheck(unittest.TestCase):

    def leaky_relu_wrapper(self, x):
        return paddle.nn.functional.leaky_relu(x[0], negative_slope=0.2)

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

        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.leaky_relu_wrapper,
                                                       [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places = [fluid.CUDAPlace(0)]
        for p in places:
            self.func(p)


class TestELUDoubleGradCheck(unittest.TestCase):

    def elu_wrapper(self, x):
        return paddle.nn.functional.elu(x[0], alpha=0.2)

    @prog_scope()
    def func(self, place):
        shape = [2, 4, 4, 4]
        eps = 1e-6
        alpha = 0.2
        dtype = np.float64
        SEED = 0

        x = layers.data('x', shape, False, dtype)
        x.persistable = True

        y = layers.elu(x, alpha=alpha)
        np.random.RandomState(SEED)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.elu_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestCELUDoubleGradCheck(unittest.TestCase):

    def celu_wrapper(self, x):
        return paddle.nn.functional.celu(x[0], alpha=0.2)

    @prog_scope()
    def func(self, place):
        shape = [2, 4, 4, 4]
        eps = 1e-6
        alpha = 0.2
        dtype = np.float64
        SEED = 0

        x = layers.data('x', shape, False, dtype)
        x.persistable = True

        y = F.celu(x, alpha=alpha)
        np.random.RandomState(SEED)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.celu_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestSqrtDoubleGradCheck(unittest.TestCase):

    def sqrt_wrapper(self, x):
        return paddle.sqrt(x[0])

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.0001
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True

        y = layers.sqrt(x)
        x_arr = np.random.uniform(0.1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.sqrt_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places = [fluid.CUDAPlace(0)]
        for p in places:
            self.func(p)


class TestRsqrtDoubleGradCheck(unittest.TestCase):

    def rsqrt_wrapper(self, x):
        return paddle.rsqrt(x[0])

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 0.0001
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True

        y = layers.rsqrt(x)
        x_arr = np.random.uniform(0.1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.rsqrt_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places = [fluid.CUDAPlace(0)]
        for p in places:
            self.func(p)


class TestSquareDoubleGradCheck(unittest.TestCase):

    def square_wrapper(self, x):
        return paddle.square(x[0])

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

        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.square_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestAbsDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 7, 9]
        eps = 1e-6
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True
        y = layers.abs(x)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        # Because we set delta = 0.005 in calculating numeric gradient,
        # if x is too small, the numeric gradient is inaccurate.
        # we should avoid this
        x_arr[np.abs(x_arr) < 0.005] = 0.02

        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestLogDoubleGradCheck(unittest.TestCase):

    def log_wrapper(self, x):
        return paddle.log(x[0])

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 7, 9]
        eps = 1e-6
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        x.persistable = True
        y = layers.log(x)

        x_arr = np.random.uniform(0.1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check([x],
                                           y,
                                           x_init=x_arr,
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.log_wrapper, [x],
                                                       y,
                                                       x_init=x_arr,
                                                       place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
