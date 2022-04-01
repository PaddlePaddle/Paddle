#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}

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

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseMinOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.random_integers(-5, 5, [10, 3, 4]).astype("float64")
        y = np.array([0.5]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.random((100, )).astype("float64")
        sgn = np.random.choice([-1, 1], (100, )).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, (100, )).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.5, 1, (100, 3, 2)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (100, )).astype(np.float64)
        y = x[:, 0, 0] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out':
            np.minimum(self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1))
        }


class TestElementwiseMinOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.5, 1, (2, 100, 3)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (100, )).astype(np.float64)
        y = x[0, :, 0] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.minimum(self.inputs['X'], self.inputs['Y'].reshape(1, 100, 1))
        }


class TestElementwiseMinOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (100, )).astype(np.float64)
        y = x[0, 0, :] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out':
            np.minimum(self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100))
        }


class TestElementwiseMinOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.5, 1, (2, 25, 4, 1)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (25, 4)).astype(np.float64)
        y = x[0, :, :, 0] + sgn * \
            np.random.uniform(1, 2, (25, 4)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.minimum(self.inputs['X'], self.inputs['Y'].reshape(1, 25, 4, 1))
        }


class TestElementwiseMinOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.5, 1, (2, 10, 2, 5)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(np.float64)
        y = x + sgn * \
            np.random.uniform(1, 2, (2, 10, 1, 5)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinOpFP16(unittest.TestCase):
    def get_out_and_grad(self, x_np, y_np, axis, place, use_fp32=False):
        assert x_np.dtype == np.float16
        assert y_np.dtype == np.float16
        if use_fp32:
            x_np = x_np.astype(np.float32)
            y_np = y_np.astype(np.float32)
        dtype = np.float16

        with fluid.dygraph.guard(place):
            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            x.stop_gradient = False
            y.stop_gradient = False
            z = fluid.layers.elementwise_min(x, y, axis)
            x_g, y_g = paddle.grad([z], [x, y])
            return z.numpy().astype(dtype), x_g.numpy().astype(
                dtype), y_g.numpy().astype(dtype)

    def check_main(self, x_shape, y_shape, axis=-1):
        if not paddle.is_compiled_with_cuda():
            return
        place = paddle.CUDAPlace(0)
        if not core.is_float16_supported(place):
            return

        x_np = np.random.random(size=x_shape).astype(np.float16)
        y_np = np.random.random(size=y_shape).astype(np.float16)

        z_1, x_g_1, y_g_1 = self.get_out_and_grad(x_np, y_np, axis, place,
                                                  False)
        z_2, x_g_2, y_g_2 = self.get_out_and_grad(x_np, y_np, axis, place, True)
        self.assertTrue(np.array_equal(z_1, z_2), "{} vs {}".format(z_1, z_2))
        self.assertTrue(
            np.array_equal(x_g_1, x_g_2), "{} vs {}".format(x_g_1, x_g_2))
        self.assertTrue(
            np.array_equal(y_g_1, y_g_2), "{} vs {}".format(y_g_1, y_g_2))

    def test_main(self):
        self.check_main((13, 17), (13, 17))
        self.check_main((10, 3, 4), (1, ))
        self.check_main((100, ), (100, ))
        self.check_main((100, 3, 2), (100, ), 0)
        self.check_main((2, 100, 3), (100, ), 1)
        self.check_main((2, 3, 100), (100, ))
        self.check_main((2, 25, 4, 1), (25, 4), 1)
        self.check_main((2, 10, 2, 5), (2, 10, 1, 5))


if __name__ == '__main__':
    unittest.main()
