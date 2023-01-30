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

<<<<<<< HEAD
import unittest

import gradient_checker
import numpy as np
from decorator_helper import prog_scope

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestElementwiseMulDoubleGradCheck(unittest.TestCase):
=======
from __future__ import print_function

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import gradient_checker

from decorator_helper import prog_scope


class TestElementwiseMulDoubleGradCheck(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.multiply(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_mul(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseMulBroadcastDoubleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._multiply_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_mul(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseAddDoubleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.add(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseAddBroadcastDoubleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._add_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseSubDoubleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def subtract_wrapper(self, x):
        return paddle.subtract(x[0], x[1])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.subtract(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.subtract_wrapper,
            [x, y],
            out,
            x_init=[x_arr, y_arr],
            place=place,
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_sub(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.subtract_wrapper,
                                                       [x, y],
                                                       out,
                                                       x_init=[x_arr, y_arr],
                                                       place=place)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseSubBroadcastDoubleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._subtract_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_sub(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseDivDoubleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def divide_wrapper(self, x):
        return paddle.divide(x[0], x[1])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.0001
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._divide_with_axis(x, y, axis=0)
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_div(x, y, axis=0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr[np.abs(y_arr) < 0.005] = 0.02

<<<<<<< HEAD
        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps, atol=1e-3
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.divide_wrapper,
            [x, y],
            out,
            x_init=[x_arr, y_arr],
            place=place,
            atol=1e-3,
        )
=======
        gradient_checker.double_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps,
                                           atol=1e-3)
        gradient_checker.double_grad_check_for_dygraph(self.divide_wrapper,
                                                       [x, y],
                                                       out,
                                                       x_init=[x_arr, y_arr],
                                                       place=place,
                                                       atol=1e-3)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseDivBroadcastDoubleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.0001
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[1:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._divide_with_axis(x, y, axis=1)
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[1:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_div(x, y, axis=1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[1:-1]).astype(dtype)
        y_arr[np.abs(y_arr) < 0.005] = 0.02

<<<<<<< HEAD
        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps, atol=1e-3
        )
=======
        gradient_checker.double_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps,
                                           atol=1e-3)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseAddTripleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.add(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.triple_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.triple_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseAddBroadcastTripleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._add_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.triple_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.triple_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseMulTripleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def multiply_wrapper(self, x):
        return paddle.multiply(x[0], x[1])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.multiply(x, y)
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_mul(x, y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.triple_grad_check(
<<<<<<< HEAD
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.multiply_wrapper,
            [x, y],
            out,
            x_init=[x_arr, y_arr],
            place=place,
        )
=======
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(
            self.multiply_wrapper, [x, y],
            out,
            x_init=[x_arr, y_arr],
            place=place)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseMulBroadcastTripleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

<<<<<<< HEAD
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._add_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.triple_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
=======
        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.triple_grad_check([x, y],
                                           out,
                                           x_init=[x_arr, y_arr],
                                           place=place,
                                           eps=eps)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
