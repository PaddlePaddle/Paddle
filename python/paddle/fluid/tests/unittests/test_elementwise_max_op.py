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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
import paddle.fluid.core as core


class TestElementwiseOp(OpTest):
=======
from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16
import os
import re
import paddle.fluid.core as core
import paddle


class TestElementwiseOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        # If x and y have the same value, the max() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}

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
<<<<<<< HEAD
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X")
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y')
        )


class TestElementwiseMaxOp_ZeroDim1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.uniform(0.1, 1, []).astype("float64")
        y = np.random.uniform(0.1, 1, []).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMaxOp_ZeroDim2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        y = np.random.uniform(0.1, 1, []).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMaxOp_ZeroDim3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.uniform(0.1, 1, []).astype("float64")
        y = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


@unittest.skipIf(
    core.is_compiled_with_cuda()
    and (
        core.cudnn_version() < 8100
        or paddle.device.cuda.get_device_capability()[0] < 8
    ),
    "run test when gpu is availble and the minimum cudnn version is 8.1.0 and gpu's compute capability is at least 8.0.",
)
class TestElementwiseBF16Op(OpTest):
=======
        self.check_grad(['Y'],
                        'Out',
                        max_relative_error=0.005,
                        no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'],
                        'Out',
                        max_relative_error=0.005,
                        no_grad_set=set('Y'))


@unittest.skipIf(core.is_compiled_with_cuda() and (
    core.cudnn_version() < 8100
    or paddle.device.cuda.get_device_capability()[0] < 8
), "run test when gpu is availble and the minimum cudnn version is 8.1.0 and gpu's compute capability is at least 8.0."
                 )
class TestElementwiseBF16Op(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        self.dtype = np.uint16
        # If x and y have the same value, the max() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(np.float32)
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        self.inputs = {
            'X': convert_float_to_uint16(x),
<<<<<<< HEAD
            'Y': convert_float_to_uint16(y),
=======
            'Y': convert_float_to_uint16(y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': convert_float_to_uint16(np.maximum(x, y))}

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
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'))


@skip_check_grad_ci(
<<<<<<< HEAD
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMaxOp_scalar(TestElementwiseOp):
=======
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseMaxOp_scalar(TestElementwiseOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.random_integers(-5, 5, [2, 3, 20]).astype("float64")
        y = np.array([0.5]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMaxOp_Vector(TestElementwiseOp):
<<<<<<< HEAD
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.random((100,)).astype("float64")
        sgn = np.random.choice([-1, 1], (100,)).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, (100,)).astype("float64")
=======

    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.random((100, )).astype("float64")
        sgn = np.random.choice([-1, 1], (100, )).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, (100, )).astype("float64")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMaxOp_broadcast_0(TestElementwiseOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.uniform(0.5, 1, (100, 5, 2)).astype(np.float64)
<<<<<<< HEAD
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float64)
        y = x[:, 0, 0] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float64
        )
=======
        sgn = np.random.choice([-1, 1], (100, )).astype(np.float64)
        y = x[:, 0, 0] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(np.float64)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 0}
        self.outputs = {
<<<<<<< HEAD
            'Out': np.maximum(
                self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1)
            )
=======
            'Out': np.maximum(self.inputs['X'],
                              self.inputs['Y'].reshape(100, 1, 1))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestElementwiseMaxOp_broadcast_1(TestElementwiseOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.uniform(0.5, 1, (2, 100, 3)).astype(np.float64)
<<<<<<< HEAD
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float64)
        y = x[0, :, 0] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float64
        )
=======
        sgn = np.random.choice([-1, 1], (100, )).astype(np.float64)
        y = x[0, :, 0] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(np.float64)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
<<<<<<< HEAD
            'Out': np.maximum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 100, 1)
            )
=======
            'Out': np.maximum(self.inputs['X'],
                              self.inputs['Y'].reshape(1, 100, 1))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestElementwiseMaxOp_broadcast_2(TestElementwiseOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.uniform(0.5, 1, (1, 3, 100)).astype(np.float64)
<<<<<<< HEAD
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float64)
        y = x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float64
        )
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out': np.maximum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100)
            )
=======
        sgn = np.random.choice([-1, 1], (100, )).astype(np.float64)
        y = x[0, 0, :] + sgn * \
            np.random.uniform(1, 2, (100, )).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out': np.maximum(self.inputs['X'],
                              self.inputs['Y'].reshape(1, 1, 100))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestElementwiseMaxOp_broadcast_3(TestElementwiseOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.uniform(0.5, 1, (2, 50, 2, 1)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (50, 2)).astype(np.float64)
<<<<<<< HEAD
        y = x[0, :, :, 0] + sgn * np.random.uniform(1, 2, (50, 2)).astype(
            np.float64
        )
=======
        y = x[0, :, :, 0] + sgn * \
            np.random.uniform(1, 2, (50, 2)).astype(np.float64)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
<<<<<<< HEAD
            'Out': np.maximum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 50, 2, 1)
            )
=======
            'Out':
            np.maximum(self.inputs['X'], self.inputs['Y'].reshape(1, 50, 2, 1))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestElementwiseMaxOp_broadcast_4(TestElementwiseOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 5)).astype(np.float64)
<<<<<<< HEAD
        y = x + sgn * np.random.uniform(1, 2, (2, 3, 1, 5)).astype(np.float64)
=======
        y = x + sgn * \
            np.random.uniform(1, 2, (2, 3, 1, 5)).astype(np.float64)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


if __name__ == '__main__':
    unittest.main()
