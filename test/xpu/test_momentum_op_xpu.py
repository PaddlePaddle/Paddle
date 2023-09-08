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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle.base import core

paddle.enable_static()


def calculate_momentum_by_numpy(
    param,
    grad,
    mu,
    velocity,
    use_nesterov,
    learning_rate,
    regularization_method,
    regularization_coeff,
):
    if regularization_method == "l2_decay":
        grad = grad + regularization_coeff * param
        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - (grad + velocity_out * mu) * learning_rate
        else:
            param_out = param - learning_rate * velocity_out
    else:
        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = (
                param - grad * learning_rate - velocity_out * mu * learning_rate
            )
        else:
            param_out = param - learning_rate * velocity_out
    return param_out, velocity_out


class XPUTestMomentumOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'momentum'
        self.use_dynamic_create_class = False

    class TestMomentumOPBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.xpu_version = core.get_xpu_device_version(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            self.op_type = 'momentum'
            self.init_config()

            self.param = np.random.uniform(-1, 1, self.input_shape).astype(
                self.dtype
            )
            self.grad = np.random.uniform(-1, 1, self.input_shape).astype(
                self.dtype
            )
            self.velocity = np.random.uniform(-1, 1, self.input_shape).astype(
                self.dtype
            )
            param_out, velocity_out = calculate_momentum_by_numpy(
                param=self.param,
                grad=self.grad,
                mu=self.mu,
                velocity=self.velocity,
                use_nesterov=self.use_nesterov,
                learning_rate=self.learning_rate,
                regularization_method=self.regularization_method,
                regularization_coeff=self.regularization_coeff,
            )
            param_out = param_out.astype(self.dtype)
            velocity_out = velocity_out.astype(self.dtype)
            self.inputs = {
                'Param': self.param,
                'Grad': self.grad,
                'Velocity': self.velocity,
                'LearningRate': self.learning_rate,
            }
            self.attrs = {
                'use_xpu': True,
                'mu': self.mu,
                'use_nesterov': self.use_nesterov,
                'regularization_method': self.regularization_method,
                'regularization_coeff': self.regularization_coeff,
            }
            self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def init_config(self):
            self.input_shape = [864]
            self.learning_rate = np.array([0.001]).astype(float)
            self.mu = 0.0001
            self.use_nesterov = False
            self.regularization_method = None
            self.regularization_coeff = 0

    class XPUTestMomentum1(TestMomentumOPBase):
        def init_config(self):
            self.input_shape = [2, 768]
            self.learning_rate = np.array([0.002]).astype(float)
            self.mu = 0.001
            self.use_nesterov = False
            self.regularization_method = None
            self.regularization_coeff = 0

    class XPUTestMomentum2(TestMomentumOPBase):
        def init_config(self):
            self.input_shape = [3, 8, 4096]
            self.learning_rate = np.array([0.005]).astype(float)
            self.mu = 0.002
            self.use_nesterov = True
            self.regularization_method = None
            self.regularization_coeff = 0

    class XPUTestMomentum3(TestMomentumOPBase):
        def init_config(self):
            self.input_shape = [1024]
            self.learning_rate = np.array([0.01]).astype(float)
            self.mu = 0.0001
            self.use_nesterov = False
            if self.xpu_version != core.XPUVersion.XPU1:
                self.regularization_method = "l2_decay"
                self.regularization_coeff = 0.005
            else:
                # regularization not supported on XPU1
                self.regularization_method = None
                self.regularization_coeff = 0

    class XPUTestMomentum4(TestMomentumOPBase):
        def init_config(self):
            self.input_shape = [2, 2, 255]
            self.learning_rate = np.array([0.0005]).astype(float)
            self.mu = 0.005
            self.use_nesterov = True
            if self.xpu_version != core.XPUVersion.XPU1:
                self.regularization_method = "l2_decay"
                self.regularization_coeff = 0.005
            else:
                # regularization not supported on XPU1
                self.regularization_method = None
                self.regularization_coeff = 0


support_types = get_xpu_op_support_types('momentum')
for stype in support_types:
    create_test_class(globals(), XPUTestMomentumOP, stype)

if __name__ == "__main__":
    unittest.main()
