#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
from paddle import base
from paddle.base import core


def pow_grad(x, y, dout):
    dx = dout * y * np.power(x, (y - 1))
    dy = dout * np.log(x) * np.power(x, y)
    return dx, dy


class TestElementwisePowOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"
        self.inputs = {
            'X': np.random.uniform(1, 2, [20, 5]).astype("float64"),
            'Y': np.random.uniform(1, 2, [20, 5]).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        if hasattr(self, 'attrs'):
            self.check_output(check_dygraph=False)
        else:
            self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        if hasattr(self, 'attrs'):
            self.check_grad(
                ['X', 'Y'], 'Out', check_prim=True, check_dygraph=False
            )
        else:
            self.check_grad(
                ['X', 'Y'],
                'Out',
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )


class TestElementwisePowOp_ZeroDim1(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(1, 2, []).astype("float64"),
            'Y': np.random.uniform(1, 2, []).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_ZeroDim2(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(1, 2, [20, 5]).astype("float64"),
            'Y': np.random.uniform(1, 2, []).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_ZeroDim3(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(1, 2, []).astype("float64"),
            'Y': np.random.uniform(1, 2, [20, 5]).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_big_shape_1(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(1, 2, [10, 10]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [10, 10]).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_big_shape_2(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(1, 2, [10, 10]).astype("float64"),
            'Y': np.random.uniform(0.2, 2, [10, 10]).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwisePowOp_scalar(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [3, 3, 4]).astype(np.float64),
            'Y': np.random.uniform(0.1, 1, [1]).astype(np.float64),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_tensor(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [100]).astype("float64"),
            'Y': np.random.uniform(1, 3, [100]).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_broadcast_0(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 1, 100]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOp_broadcast_4(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 10, 3, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 10, 1, 5]).astype("float64"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


class TestElementwisePowOpInt(OpTest):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {'X': np.asarray([1, 3, 6]), 'Y': np.asarray([1, 1, 1])}
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        if hasattr(self, 'attrs'):
            self.check_output(check_dygraph=False)
        else:
            self.check_output(check_pir=True, check_symbol_infer=False)


class TestElementwisePowGradOpInt(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray([1, 3, 6])
        self.y = np.asarray([1, 1, 1])
        self.res = self.x**self.y

        # dout = 1
        self.grad_res = np.asarray([1, 1, 1])
        # dx = dout * y * pow(x, y-1)
        self.grad_x = (
            self.grad_res * self.y * (self.x ** (self.y - 1)).astype("int")
        )
        # dy = dout * log(x) * pow(x, y)
        self.grad_y = (
            self.grad_res * np.log(self.x) * (self.x**self.y)
        ).astype("int")

    def test_grad(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if base.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            with base.dygraph.guard(place):
                x = paddle.to_tensor(self.x)
                y = paddle.to_tensor(self.y)
                x.stop_gradient = False
                y.stop_gradient = False
                res = x**y
                res.retain_grads()
                res.backward()
                np.testing.assert_array_equal(res.gradient(), self.grad_res)
                np.testing.assert_array_equal(x.gradient(), self.grad_x)
                np.testing.assert_array_equal(y.gradient(), self.grad_y)


class TestElementwisePowOpFP16(OpTest):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.dtype = np.float16
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"

        self.inputs = {
            'X': np.random.uniform(1, 2, [20, 5]).astype("float16"),
            'Y': np.random.uniform(1, 2, [20, 5]).astype("float16"),
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        if hasattr(self, 'attrs'):
            self.check_output(check_dygraph=False)
        else:
            self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=pow_grad(
                self.inputs['X'], self.inputs['Y'], 1 / self.inputs['X'].size
            ),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
    "BFP16 test runs only on CUDA",
)
class TestElementwisePowBF16Op(OpTest):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.prim_op_type = "prim"
        self.dtype = np.uint16
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow

        x = np.random.uniform(0, 1, [20, 5]).astype(np.float32)
        y = np.random.uniform(0, 1, [20, 5]).astype(np.float32)
        out = np.power(x, y)
        self.inputs = {
            'X': convert_float_to_uint16(x),
            'Y': convert_float_to_uint16(y),
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(
                core.CUDAPlace(0),
                ['X', 'Y'],
                'Out',
                check_prim=True,
                only_check_prim=True,
                check_prim_pir=True,
            )


if __name__ == '__main__':
    unittest.main()
