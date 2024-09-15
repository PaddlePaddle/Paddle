# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest

sys.path.append("../../../legacy_test")
import gradient_checker
import numpy as np
import parameterized as param
from decorator_helper import prog_scope
from utils import dygraph_guard

import paddle
from paddle import base
from paddle.base import core


@param.parameterized_class(
    ('shape1', 'shape2'),
    [
        (
            [2, 3, 4],
            [2, 3, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 1],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 1],
        ),
    ],
)
class TestAddHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def add_wrapper(self, x):
        return paddle.add(x[0], x[1])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.add(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check(
            [x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.add_wrapper, [x, y], y=out, x_init=[x_arr, y_arr], place=place
        )
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.add(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check(
            [x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.add_wrapper, [x, y], y=out, x_init=[x_arr, y_arr], place=place
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)


@param.parameterized_class(
    ('shape1', 'shape2'),
    [
        (
            [2, 3, 4],
            [2, 3, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 1],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 1],
        ),
    ],
)
class TestSubtractHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def subtract_wrapper(self, x):
        return paddle.subtract(x[0], x[1])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.subtract(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check(
            [x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.subtract_wrapper,
            [x, y],
            y=out,
            x_init=[x_arr, y_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.subtract(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check(
            [x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.subtract_wrapper,
            [x, y],
            y=out,
            x_init=[x_arr, y_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)


@param.parameterized_class(
    ('shape1', 'shape2'),
    [
        (
            [2, 3, 4],
            [2, 3, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 1],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 1],
        ),
    ],
)
class TestMultiplyHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def multiply_wrapper(self, x):
        return paddle.multiply(x[0], x[1])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.multiply(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check(
            [x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.multiply_wrapper,
            [x, y],
            y=out,
            x_init=[x_arr, y_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.multiply(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check(
            [x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.multiply_wrapper,
            [x, y],
            y=out,
            x_init=[x_arr, y_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)


@param.parameterized_class(
    ('shape1'),
    [
        ([2],),
        ([2, 3],),
        ([2, 3, 4],),
        ([2, 3, 3, 4],),
    ],
)
class TestSiluHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1

    def silu_wrapper(self, x):
        return paddle.nn.functional.silu(x[0])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.nn.functional.silu(x)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002

        # silu double grad only has CompositeOpMaker,don't need set prim_flag
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check(
            [x], y=out, x_init=[x_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.silu_wrapper,
            [x],
            y=out,
            x_init=[x_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.nn.functional.silu(x)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check(
            [x], y=out, x_init=[x_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.silu_wrapper,
            [x],
            y=out,
            x_init=[x_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)


@param.parameterized_class(
    ('shape1'),
    [
        ([2],),
        ([2, 3],),
        ([2, 3, 4],),
        ([2, 3, 3, 4],),
    ],
)
class TestWhereHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1

    def _where_wrapper(self, x):
        return paddle.where(x[0], x[1], x[2])

    @prog_scope()
    def _func_double(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64

        cond = paddle.static.data('cond', shape1, dtype="bool")
        cond.stop_gradient = True
        cond.persistable = True

        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True

        y = paddle.static.data('y', shape1, dtype=dtype)
        y.stop_gradient = False
        y.persistable = True

        out = paddle.where(cond, x, y)

        # generate random data
        cond_arr = np.random.choice([False, True], size=shape1)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape1).astype(dtype)

        # where double grad only has CompositeOpMaker,don't need set prim_flag
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check(
            [cond, x, y],
            y=out,
            x_init=[cond_arr, x_arr, y_arr],
            place=place,
            eps=eps,
        )
        gradient_checker.double_grad_check_for_dygraph(
            self._where_wrapper,
            [cond, x, y],
            y=out,
            x_init=[cond_arr, x_arr, y_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def _func_triple(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64

        cond = paddle.static.data('cond', shape1, dtype="bool")
        cond.stop_gradient = True
        cond.persistable = True

        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True

        y = paddle.static.data('y', shape1, dtype=dtype)
        y.stop_gradient = False
        y.persistable = True

        out = paddle.where(cond, x, y)

        # generate random data
        cond_arr = np.random.choice([False, True], size=shape1)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape1).astype(dtype)

        # where double grad only has CompositeOpMaker,don't need set prim_flag
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check(
            [cond, x, y],
            y=out,
            x_init=[cond_arr, x_arr, y_arr],
            place=place,
            eps=eps,
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.silu_wrapper,
            [cond, x, y],
            y=out,
            x_init=[cond_arr, x_arr, y_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self._func_double(p)
            self._func_triple(p)


@param.parameterized_class(
    ('shape1'),
    [
        ([2],),
        ([2, 3],),
        ([2, 3, 4],),
        ([2, 3, 3, 4],),
    ],
)
class TestExpHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1

    def exp_wrapper(self, x):
        return paddle.exp(x[0])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.exp(x)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002

        # exp double grad only has CompositeOpMaker, don't need set prim_flag
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check(
            [x], y=out, x_init=[x_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.exp_wrapper,
            [x],
            y=out,
            x_init=[x_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.exp(x)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check(
            [x], y=out, x_init=[x_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.exp_wrapper,
            [x],
            y=out,
            x_init=[x_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)


@param.parameterized_class(
    ('shape1'),
    [
        ([2],),
        ([2, 3],),
        ([2, 3, 4],),
        ([2, 3, 3, 4],),
    ],
)
class TestLogHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1

    def log_wrapper(self, x):
        return paddle.log(x[0])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.log(x)
        x_arr = np.random.uniform(0.0, 10.0, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002

        # log double grad only has CompositeOpMaker,don't need set prim_flag
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check(
            [x], y=out, x_init=[x_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.log_wrapper,
            [x],
            y=out,
            x_init=[x_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.log(x)
        x_arr = np.random.uniform(0.0, 10.0, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check(
            [x], y=out, x_init=[x_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.log_wrapper,
            [x],
            y=out,
            x_init=[x_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)


@param.parameterized_class(
    ('shape1'),
    [
        ([2],),
        ([2, 3],),
        ([2, 3, 4],),
        ([2, 3, 3, 4],),
    ],
)
class TestAbsHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1

    def abs_wrapper(self, x):
        return paddle.abs(x[0])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.abs(x)
        x_arr = np.random.uniform(0.0, 10.0, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002

        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check(
            [x], y=out, x_init=[x_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.abs_wrapper,
            [x],
            y=out,
            x_init=[x_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.abs(x)
        x_arr = np.random.uniform(0.0, 10.0, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check(
            [x], y=out, x_init=[x_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.abs_wrapper,
            [x],
            y=out,
            x_init=[x_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)


@param.parameterized_class(
    ('shape1', 'shape2'),
    [
        (
            [2, 3, 4],
            [2, 3, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 1],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 1],
        ),
    ],
)
class TestMinimumHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def minimum_wrapper(self, x):
        return paddle.minimum(x[0], x[1])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.minimum(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        core._set_prim_backward_blacklist("minimum_grad")
        gradient_checker.double_grad_check(
            [x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.minimum_wrapper,
            [x, y],
            y=out,
            x_init=[x_arr, y_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)


@param.parameterized_class(
    ('shape1', 'shape2'),
    [
        (
            [2, 3, 4],
            [2, 3, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [3, 1, 1],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 4],
        ),
        (
            [2, 3, 3, 4],
            [2, 3, 1, 1],
        ),
    ],
)
class TestMaximumHighGradCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def maximum_wrapper(self, x):
        return paddle.maximum(x[0], x[1])

    @prog_scope()
    def func_double(self, place):
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.maximum(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core

        core._set_prim_backward_enabled(True)
        core._set_prim_backward_blacklist("minimum_grad")
        gradient_checker.double_grad_check(
            [x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.maximum_wrapper,
            [x, y],
            y=out,
            x_init=[x_arr, y_arr],
            place=place,
        )
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)


@param.parameterized_class(
    ('shape1', 'shape2'),
    [
        ([2], [2], True),
        ([2, 3], [2, 3], True),
        ([2, 3, 4], [2, 3, 4], True),
        ([2, 3, 3, 4], [2, 3, 3, 4], True),
    ],
)
class TestMaximumHighGradCheck2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def _grad(self, y, x, order):
        u = y
        dx = paddle.ones_like(x)
        for _ in range(order):
            dx = paddle.grad(u, x, create_graph=True)[0]
            u = dx
        return dx

    def func_double(self, place, x_stop, y_stop):
        x = paddle.randn(self.shape1).astype("float32")
        x.stop_gradient = x_stop
        y = paddle.randn(self.shape2).astype("float32")
        y.stop_gradient = y_stop

        # wraping with tanh to enable high order gradient
        z = paddle.maximum(paddle.tanh(x), paddle.tanh(y))

        if not x.stop_gradient:
            dzdx = self._grad(z, x, 2)
        if not y.stop_gradient:
            dzdy = self._grad(z, y, 2)

    def func_triple(self, place, x_stop, y_stop):
        x = paddle.randn(self.shape1).astype("float32")
        x.stop_gradient = x_stop
        y = paddle.randn(self.shape2).astype("float32")
        y.stop_gradient = y_stop

        # wraping with tanh to enable high order gradient
        z = paddle.maximum(paddle.tanh(x), paddle.tanh(y))

        if not x.stop_gradient:
            dzdx = self._grad(z, x, 3)
        if not y.stop_gradient:
            dzdy = self._grad(z, y, 3)

    def test_high_grad(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            for x_stop in [False, True]:
                for y_stop in [False, True]:
                    with dygraph_guard():
                        self.func_double(p, x_stop, y_stop)
                        self.func_triple(p, x_stop, y_stop)


if __name__ == '__main__':
    unittest.main()
