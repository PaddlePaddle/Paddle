#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

sys.path.append("..")

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator

paddle.enable_static()


def calculate_rmsprop_by_numpy(
    param, grad, mean_square, moment, learning_rate, epsilon, decay, momentum
):
    mean_square_out = decay * mean_square + (1 - decay) * grad * grad
    moment_out = momentum * moment + learning_rate * grad / np.sqrt(
        mean_square_out + epsilon
    )
    param_out = param - moment_out
    return param_out, mean_square_out, moment_out


class XPUTestRMSPropOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'rmsprop'
        self.use_dynamic_create_class = False

    class TestRMSPropOPBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.xpu_version = core.get_xpu_device_version(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            self.op_type = 'rmsprop'
            self.dtype = self.in_type
            self.init_config()

            self.param = np.random.uniform(-1, 1, self.input_shape).astype(
                self.dtype
            )
            self.grad = np.random.uniform(-1, 1, self.input_shape).astype(
                self.dtype
            )
            self.mean_square = np.random.uniform(0, 1, self.input_shape).astype(
                self.dtype
            )
            self.moment = np.random.uniform(-1, 1, self.input_shape).astype(
                self.dtype
            )

            self.mean_grad = np.random.uniform(-1, 1, self.input_shape).astype(
                self.dtype
            )
            self.mean_grad_out = np.random.uniform(
                -1, 1, self.input_shape
            ).astype(self.dtype)

            param_out, mean_square_out, moment_out = calculate_rmsprop_by_numpy(
                param=self.param,
                grad=self.grad,
                mean_square=self.mean_square,
                moment=self.moment,
                learning_rate=self.learning_rate,
                epsilon=self.epsilon,
                decay=self.decay,
                momentum=self.momentum,
            )
            self.inputs = {
                'Param': self.param,
                'Grad': self.grad,
                'MeanSquare': self.mean_square,
                'Moment': self.moment,
                'LearningRate': self.learning_rate,
                'MeanGrad': self.mean_grad,
                'MeanGradOut': self.mean_grad_out,
            }
            self.attrs = {
                'use_xpu': True,
                'epsilon': self.epsilon,
                'decay': self.decay,
                'momentum': self.momentum,
                'centered': False,  # TODO(houj04): when XDNN api supports 'center = True', add more test cases
            }
            self.outputs = {
                'ParamOut': param_out,
                'MomentOut': moment_out,
                'MeanSquareOut': mean_square_out,
                'MeanGradOut': self.mean_grad_out,
            }

        def init_dtype(self):
            self.dtype = np.float32

        def test_check_output(self):
            self.check_output_with_place(
                self.place, no_check_set=['MeanGradOut']
            )

        def init_config(self):
            self.input_shape = [864]
            self.learning_rate = np.array([0.001]).astype(self.dtype)
            self.epsilon = 1e-4
            self.decay = 0.9
            self.momentum = 0.1

    class XPUTestRMSProp1(TestRMSPropOPBase):
        def init_config(self):
            self.input_shape = [2, 768]
            self.learning_rate = np.array([0.002]).astype(self.dtype)
            self.epsilon = 1e-4
            self.decay = 0.9
            self.momentum = 0.1

    class XPUTestRMSProp2(TestRMSPropOPBase):
        def init_config(self):
            self.input_shape = [3, 8, 4096]
            self.learning_rate = np.array([0.005]).astype(self.dtype)
            self.epsilon = 1e-6
            self.decay = 0.95
            self.momentum = 0

    class XPUTestRMSProp3(TestRMSPropOPBase):
        def init_config(self):
            self.input_shape = [1024]
            self.learning_rate = np.array([0.01]).astype(self.dtype)
            self.epsilon = 1e-5
            self.decay = 0.99
            self.momentum = 0.02

    class XPUTestRMSProp4(TestRMSPropOPBase):
        def init_config(self):
            self.input_shape = [2, 2, 255]
            self.learning_rate = np.array([0.0005]).astype(self.dtype)
            self.epsilon = 1e-3
            self.decay = 0.8
            self.momentum = 0.002


class TestBase(unittest.TestCase):
    def setup(
        self, place, is_sparse, centered, size, row_num=None, epsilon=1e-6
    ):
        np.random.seed(5)  # fix seed

        self.scope = fluid.global_scope()
        self.place = place

        self.param_name = "param"
        self.param = np.random.random(size).astype("float32")

        self.mean_square_name = "mean_square"
        self.mean_square = np.random.uniform(low=1, high=2, size=size).astype(
            "float32"
        )

        self.mean_grad_name = "mean_grad"
        self.mean_grad = np.random.random(size).astype("float32")

        self.lr_name = "lr"
        self.learning_rate = np.array([0.01]).astype("float32")

        self.grad_name = "grad"

        self.is_sparse = is_sparse
        self.grad = np.random.random(size).astype("float32")
        grad_tensor = self.scope.var(self.grad_name).get_tensor()
        grad_tensor.set(self.grad, place)

        self.moment_name = "moment"
        self.moment = np.random.uniform(low=0, high=1, size=size).astype(
            "float32"
        )

        self.epsilon = epsilon
        self.decay = 0.9
        self.momentum = 0.1
        self.centered = centered

        self.ms_out = (
            self.decay * self.mean_square
            + (1 - self.decay) * self.grad * self.grad
        )
        if centered:
            self.mg_out = (
                self.decay * self.mean_grad + (1 - self.decay) * self.grad
            )
            self.moment_out = (
                self.momentum * self.moment
                + self.learning_rate
                * self.grad
                / np.sqrt(self.ms_out - np.square(self.mg_out) + self.epsilon)
            )
        else:
            self.moment_out = (
                self.momentum * self.moment
                + self.learning_rate
                * self.grad
                / np.sqrt(self.ms_out + self.epsilon)
            )

        self.param_out = self.param - self.moment_out

        # create and initialize Param Variable
        self.param_tensor = self.scope.var(self.param_name).get_tensor()
        self.param_tensor.set(self.param, place)

        self.mean_square_tensor = self.scope.var(
            self.mean_square_name
        ).get_tensor()
        self.mean_square_tensor.set(self.mean_square, place)

        lr = self.scope.var(self.lr_name).get_tensor()
        lr.set(self.learning_rate, place)

        self.moment_tensor = self.scope.var(self.moment_name).get_tensor()
        self.moment_tensor.set(self.moment, place)

        if self.centered:
            self.mean_grad_tensor = self.scope.var(
                self.mean_grad_name
            ).get_tensor()
            self.mean_grad_tensor.set(self.mean_grad, place)

    def check(self, actual_t, expect_t, place, out_name, atol=1e-5):
        np.testing.assert_allclose(
            actual_t,
            expect_t,
            rtol=1e-05,
            atol=atol,
            err_msg='Output ('
            + out_name
            + ') has diff at '
            + str(place)
            + '\nExpect '
            + str(expect_t)
            + '\n'
            + 'But Got'
            + str(actual_t),
        )


class TestRmspropOp(TestBase):
    def check_with_place(
        self, place, is_sparse, centered, size, row_num=None, epsilon=1e-6
    ):
        self.setup(place, is_sparse, centered, size, row_num, epsilon)
        self.run_and_check()

    def run_and_check(self):
        grad_name = self.grad_name

        kwargs = {
            'Param': self.param_name,
            'Grad': grad_name,
            'MeanSquare': self.mean_square_name,
            'Moment': self.moment_name,
            'LearningRate': self.lr_name,
            'ParamOut': self.param_name,
            'MeanSquareOut': self.mean_square_name,
            'MomentOut': self.moment_name,
            'epsilon': self.epsilon,
            'decay': self.decay,
            'momentum': self.momentum,
            'centered': self.centered,
        }

        if self.centered:
            kwargs['MeanGrad'] = self.mean_grad_name
            kwargs['MeanGradOut'] = self.mean_grad_name

        rmsprop_op = Operator('rmsprop', **kwargs)
        atol = 1e-6

        rmsprop_op.run(self.scope, self.place)

        self.check(
            np.array(self.mean_square_tensor),
            self.ms_out,
            self.place,
            self.mean_square_name,
            atol=atol,
        )
        self.check(
            np.array(self.moment_tensor),
            self.moment_out,
            self.place,
            self.moment_name,
            atol=atol,
        )
        self.check(
            np.array(self.param_tensor),
            self.param_out,
            self.place,
            self.param_name,
            atol=atol,
        )

        if self.centered:
            self.check(
                np.array(self.mean_grad_tensor),
                self.mg_out,
                self.place,
                self.mean_grad_name,
            )

    def test_rmsprop(self):
        places = [core.XPUPlace(0)]

        size = (128, 320)
        for place in places:
            for centered in [False, True]:
                with fluid.scope_guard(core.Scope()):
                    self.check_with_place(
                        place, is_sparse=False, centered=centered, size=size
                    )


support_types = get_xpu_op_support_types('rmsprop')
for stype in support_types:
    create_test_class(globals(), XPUTestRMSPropOP, stype)

if __name__ == "__main__":
    unittest.main()
