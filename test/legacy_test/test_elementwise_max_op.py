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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
from paddle.base import core


class TestElementwiseOp(OpTest):
    def init_data(self):
        # If x and y have the same value, the max() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float64")
        self.y = self.x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(
            "float64"
        )

    def setUp(self):
        self.init_data()
        self.op_type = "elementwise_max"
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self.python_api = paddle.maximum
        self.public_python_api = paddle.maximum
        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        if hasattr(self, 'attrs'):
            self.check_output(check_dygraph=False)
        else:
            self.check_output()

    def test_check_grad_normal(self):
        if hasattr(self, 'attrs'):
            if self.attrs['axis'] == -1:
                self.check_grad(
                    ['X', 'Y'],
                    'Out',
                    check_dygraph=False,
                    check_prim=True,
                    check_prim_pir=True,
                )
            else:
                self.check_grad(['X', 'Y'], 'Out', check_dygraph=False)
        else:
            self.check_grad(
                ['X', 'Y'], 'Out', check_prim=True, check_prim_pir=True
            )

    def test_check_grad_ignore_x(self):
        if hasattr(self, 'attrs') and self.attrs['axis'] != -1:
            self.check_grad(
                ['Y'],
                'Out',
                max_relative_error=0.005,
                no_grad_set=set("X"),
                check_dygraph=False,
            )
        else:
            self.check_grad(
                ['Y'],
                'Out',
                max_relative_error=0.005,
                no_grad_set=set("X"),
                check_prim=True,
                check_prim_pir=True,
            )

    def test_check_grad_ignore_y(self):
        if hasattr(self, 'attrs') and self.attrs['axis'] != -1:
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.005,
                no_grad_set=set('Y'),
                check_dygraph=False,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.005,
                no_grad_set=set('Y'),
                check_prim=True,
                check_prim_pir=True,
            )

    def if_enable_cinn(self):
        pass


class TestElementwiseFP16Op(TestElementwiseOp):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float16)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(np.float16)
        self.y = self.x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(
            np.float16
        )

    def setUp(self):
        self.init_data()
        self.op_type = "elementwise_max"
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self.python_api = paddle.maximum
        self.dtype = np.float16
        self.public_python_api = paddle.maximum
        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMaxOp_ZeroDim1(TestElementwiseOp):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float64")
        self.y = np.random.uniform(0.1, 1, []).astype("float64")


class TestElementwiseMaxFP16Op_ZeroDim1(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype(np.float16)
        self.y = np.random.uniform(0.1, 1, []).astype(np.float16)


class TestElementwiseMaxOp_ZeroDim2(TestElementwiseOp):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.y = np.random.uniform(0.1, 1, []).astype("float64")


class TestElementwiseMaxFP16Op_ZeroDim2(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float16)
        self.y = np.random.uniform(0.1, 1, []).astype(np.float16)


class TestElementwiseMaxOp_ZeroDim3(TestElementwiseOp):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float64")
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype("float64")


class TestElementwiseMaxFP16Op_ZeroDim3(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype(np.float16)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(np.float16)


@unittest.skipIf(
    core.is_compiled_with_cuda()
    and (
        core.cudnn_version() < 8100
        or paddle.device.cuda.get_device_capability()[0] < 8
    ),
    "run test when gpu is available and the minimum cudnn version is 8.1.0 and gpu's compute capability is at least 8.0.",
)
class TestElementwiseBF16Op(OpTest):
    def init_data(self):
        # If x and y have the same value, the max() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(np.float32)
        self.y = self.x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(
            np.float32
        )

    def setUp(self):
        self.init_data()
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        self.public_python_api = paddle.maximum
        self.prim_op_type = "prim"
        self.dtype = np.uint16
        self.inputs = {
            'X': convert_float_to_uint16(self.x),
            'Y': convert_float_to_uint16(self.y),
        }
        self.outputs = {
            'Out': convert_float_to_uint16(np.maximum(self.x, self.y))
        }
        self.if_enable_cinn()

    def test_check_output(self):
        if hasattr(self, 'attrs'):
            self.check_output(check_dygraph=False)
        else:
            self.check_output(check_dygraph=True)

    def if_enable_cinn(self):
        pass

    def test_check_grad_normal(self):
        if hasattr(self, 'attrs'):
            # check_prim=False, bfloat16 is not supported in `less_equal`
            self.check_grad(
                ['X', 'Y'], 'Out', numeric_grad_delta=0.05, check_dygraph=False
            )
        else:
            self.check_grad(
                ['X', 'Y'],
                'Out',
                numeric_grad_delta=0.05,
                check_prim=True,
                check_prim_pir=True,
            )

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            numeric_grad_delta=0.05,
            no_grad_set=set("X"),
            check_prim=True,
            check_prim_pir=True,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            numeric_grad_delta=0.05,
            no_grad_set=set('Y'),
            check_prim=True,
            check_prim_pir=True,
        )


class TestElementwiseMaxBF16Op_ZeroDim1(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float32")
        self.y = np.random.uniform(0.1, 1, []).astype("float32")


class TestElementwiseMaxBF16Op_scalar(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.random_integers(-5, 5, [2, 3, 20]).astype("float32")
        self.y = np.array([0.5]).astype("float32")
        self.__class__.no_need_check_grad = True


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMaxOp_scalar(TestElementwiseOp):
    def init_data(self):
        self.x = np.random.random_integers(-5, 5, [2, 3, 20]).astype("float64")
        self.y = np.array([0.5]).astype("float64")


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMaxFP16Op_scalar(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.random_integers(-5, 5, [2, 3, 20]).astype(np.float16)
        self.y = np.array([0.5]).astype(np.float16)


class TestElementwiseMaxOp_Vector(TestElementwiseOp):
    def init_data(self):
        self.x = np.random.random((100,)).astype("float64")
        sgn = np.random.choice([-1, 1], (100,)).astype("float64")
        self.y = self.x + sgn * np.random.uniform(0.1, 1, (100,)).astype(
            "float64"
        )


class TestElementwiseMaxFP16Op_Vector(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.random((100,)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float16)
        self.y = self.x + sgn * np.random.uniform(0.1, 1, (100,)).astype(
            np.float16
        )


class TestElementwiseMaxBF16Op_Vector(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.random((100,)).astype("float32")
        sgn = np.random.choice([-1, 1], (100,)).astype("float32")
        self.y = self.x + sgn * np.random.uniform(0.1, 1, (100,)).astype(
            "float32"
        )


class TestElementwiseMaxOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        self.public_python_api = paddle.maximum
        self.prim_op_type = "prim"
        x = np.random.uniform(0.5, 1, (1, 3, 100)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float64)
        y = x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float64
        )
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out': np.maximum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100)
            )
        }


class TestElementwiseMaxFP16Op_broadcast_2(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        self.public_python_api = paddle.maximum
        self.prim_op_type = "prim"
        self.dtype = np.float16
        x = np.random.uniform(0.5, 1, (1, 3, 100)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float16)
        y = x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float16
        )
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out': np.maximum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100)
            )
        }


class TestElementwiseMaxOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        self.public_python_api = paddle.maximum
        self.prim_op_type = "prim"
        x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 5)).astype(np.float64)
        y = x + sgn * np.random.uniform(1, 2, (2, 3, 1, 5)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseFP16Op_broadcast_4(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        self.public_python_api = paddle.maximum
        self.prim_op_type = "prim"
        self.dtype = np.float16
        x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 5)).astype(np.float16)
        y = x + sgn * np.random.uniform(1, 2, (2, 3, 1, 5)).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


if __name__ == '__main__':
    unittest.main()
