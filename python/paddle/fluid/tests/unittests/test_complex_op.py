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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest

import paddle
from paddle.fluid import dygraph
from paddle import static
paddle.enable_static()


def ref_complex(x, y):
    return x + 1j * y


def ref_complex_grad(x, y, dout):
    out = x + 1j * y
    out_rank = out.ndim
    delta_rank_x = out_rank - x.ndim
    delta_rank_y = out_rank - y.ndim

    dx_reduce_axes = []
    dy_reduce_axes = []

    for i in range(out_rank):
        if i < delta_rank_x or dout.shape[i] > x.shape[i - delta_rank_x]:
            dx_reduce_axes.append(i)
        if i < delta_rank_y or dout.shape[i] > y.shape[i - delta_rank_y]:
            dy_reduce_axes.append(i)
    dx = np.sum(dout.real, axis=tuple(dx_reduce_axes)).reshape(x.shape)
    dy = np.sum(dout.imag, axis=tuple(dy_reduce_axes)).reshape(y.shape)
    return (dx, dy)


class TestComplexOp(OpTest):
    def init_spec(self):
        self.x_shape = [10, 10]
        self.y_shape = [10, 10]
        self.dtype = "float64"

    def setUp(self):
        self.op_type = "complex"
        self.init_spec()
        x = np.random.randn(*self.x_shape).astype(self.dtype)
        y = np.random.randn(*self.y_shape).astype(self.dtype)
        out_ref = ref_complex(x, y)
        self.out_grad = np.random.randn(*self.x_shape).astype(self.dtype) \
                      + 1j * np.random.randn(*self.y_shape).astype(self.dtype)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        dout = self.out_grad
        dx, dy = ref_complex_grad(self.inputs['X'], self.inputs['Y'],
                                  self.out_grad)
        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=[dx, dy],
            user_defined_grad_outputs=[dout])

    def test_check_grad_ignore_x(self):
        dout = self.out_grad
        dx, dy = ref_complex_grad(self.inputs['X'], self.inputs['Y'],
                                  self.out_grad)
        self.assertTupleEqual(dx.shape, tuple(self.x_shape))
        self.assertTupleEqual(dy.shape, tuple(self.y_shape))
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set('X'),
            user_defined_grads=[dy],
            user_defined_grad_outputs=[dout])

    def test_check_grad_ignore_y(self):
        dout = self.out_grad
        dx, dy = ref_complex_grad(self.inputs['X'], self.inputs['Y'],
                                  self.out_grad)
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            user_defined_grads=[dx],
            user_defined_grad_outputs=[dout])


class TestComplexOpBroadcast1(TestComplexOp):
    def init_spec(self):
        self.x_shape = [10, 3, 1, 4]
        self.y_shape = [100, 1]
        self.dtype = "float64"


class TestComplexOpBroadcast2(TestComplexOp):
    def init_spec(self):
        self.x_shape = [100, 1]
        self.y_shape = [10, 3, 1, 4]
        self.dtype = "float32"


class TestComplexOpBroadcast3(TestComplexOp):
    def init_spec(self):
        self.x_shape = [1, 100]
        self.y_shape = [100]
        self.dtype = "float32"


class TestComplexAPI(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(10, 10)
        self.y = np.random.randn(10, 10)
        self.out = ref_complex(self.x, self.y)

    def test_dygraph(self):
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            y = paddle.to_tensor(self.y)
            out_np = paddle.complex(x, y).numpy()
        self.assertTrue(np.allclose(self.out, out_np))

    def test_static(self):
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x = static.data("x", shape=[10, 10], dtype="float64")
            y = static.data("y", shape=[10, 10], dtype="float64")
            out = paddle.complex(x, y)

        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(mp,
                           feed={"x": self.x,
                                 "y": self.y},
                           fetch_list=[out])
        self.assertTrue(np.allclose(self.out, out_np))


if __name__ == "__main__":
    unittest.main()
