#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
# import paddle.fluid as fluid
import paddle.static as static
import paddle


class TestElementwiseHeavisideOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-10, 20, [20, 5]).astype("float64"),
            'Y': np.random.uniform(-11, 9, [20, 5]).astype("float64")
        }
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        if hasattr(self, 'attrs'):
            self.check_output(check_eager=False)
        else:
            self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        if hasattr(self, 'attrs'):
            self.check_grad(['X', 'Y'], 'Out', check_eager=False)
        else:
            self.check_grad(['X', 'Y'], 'Out', check_eager=True)


class TestElementwiseHeavisideOp_big_shape_1(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-10, 20, [50, 40, 3]).astype("float64"),
            'Y': np.random.uniform(-10, 11, [50, 40, 3]).astype("float64")
        }
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseHeavisideOp_big_shape_2(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-1, 2, [10, 10, 20]).astype("float64"),
            'Y': np.random.uniform(-3, 4, [10, 10, 20]).astype("float64")
        }
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseHeavisideOp_scalar(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-10, 10, [3, 3, 4]).astype(np.float64),
            'Y': np.random.uniform(-10, 10, [1]).astype(np.float64)
        }
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseHeavisideOp_tensor(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-10, 50, [100]).astype("float64"),
            'Y': np.random.uniform(-15, 37, [100]).astype("float64")
        }
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseHeavisideOp_broadcast_0(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-20, 20, [2, 1, 100]).astype("float64"),
            'Y': np.random.uniform(-30, 1, [100]).astype("float64")
        }
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseHeavisideOp_broadcast_1(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-100, 100, [2, 100, 1]).astype("float64"),
            'Y': np.random.uniform(-99, -1, [100]).astype("float64")
        }
        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.heaviside(self.inputs['X'], self.inputs['Y'].reshape(100, 1))
        }


class TestElementwiseHeavisideOp_broadcast_2(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-8, 1, [100, 3, 1]).astype("float64"),
            'Y': np.random.uniform(100, 200, [100]).astype("float64")
        }
        self.attrs = {'axis': 0}
        self.outputs = {
            'Out':
            np.heaviside(self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1))
        }


class TestElementwiseHeavisideOp_broadcast_3(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-9.1, 10, [2, 20, 5, 1]).astype("float64"),
            'Y': np.random.uniform(8.1, 11, [20, 5]).astype("float64")
        }
        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.heaviside(self.inputs['X'], self.inputs['Y'].reshape(1, 20, 5, 1))
        }


class TestElementwiseHeavisideOp_broadcast_4(TestElementwiseHeavisideOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(-10, 10, [2, 10, 3, 5]).astype("float64"),
            'Y': np.random.uniform(-10, 10, [2, 10, 1, 5]).astype("float64")
        }
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseHeavisideOpInt(OpTest):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {'X': np.asarray([0, -3, 6]), 'Y': np.asarray([1, 1, 1])}
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        if hasattr(self, 'attrs'):
            self.check_output(check_eager=False)
        else:
            self.check_output(check_eager=True)


# class TestElementwisePowGradOpInt(unittest.TestCase):
#     def setUp(self):
#         self.x = np.asarray([1, 3, 6])
#         self.y = np.asarray([1, 1, 1])
#         self.res = self.x**self.y
#         # dout = 1
#         self.grad_res = np.asarray([1, 1, 1])
#         # dx = dout * y * pow(x, y-1)
#         self.grad_x = self.grad_res * self.y * (self.x
#                                                 **(self.y - 1)).astype("int")
#         # dy = dout * log(x) * pow(x, y)
#         self.grad_y = (self.grad_res * np.log(self.x) *
#                        (self.x**self.y)).astype("int")
#         print(self.grad_res, self.grad_x, self.grad_y)

#     def test_grad(self):
#         places = [fluid.CPUPlace()]
#         if fluid.is_compiled_with_cuda():
#             places.append(fluid.CUDAPlace(0))
#         for place in places:
#             with fluid.dygraph.guard(place):
#                 x = fluid.dygraph.to_variable(self.x, zero_copy=False)
#                 y = fluid.dygraph.to_variable(self.y, zero_copy=False)
#                 print(x, y)
#                 x.stop_gradient = False
#                 y.stop_gradient = False
#                 res = x**y
#                 res.backward()
#                 self.assertTrue(np.array_equal(res.gradient(), self.grad_res))
#                 self.assertTrue(np.array_equal(x.gradient(), self.grad_x))
#                 self.assertTrue(np.array_equal(y.gradient(), self.grad_y))


if __name__ == '__main__':
    unittest.main()
