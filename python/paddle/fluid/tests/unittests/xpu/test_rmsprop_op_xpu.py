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
import sys

sys.path.append("..")

import paddle
import paddle.fluid.core as core

from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


def calculate_rmsprop_by_numpy(param, grad, mean_square, moment, learning_rate,
                               epsilon, decay, momentum):
    mean_square_out = decay * mean_square + (1 - decay) * grad * grad
    moment_out = momentum * moment + learning_rate * grad / np.sqrt(
        mean_square_out + epsilon)
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

            self.param = np.random.uniform(-1, 1,
                                           self.input_shape).astype(self.dtype)
            self.grad = np.random.uniform(-1, 1,
                                          self.input_shape).astype(self.dtype)
            self.mean_square = np.random.uniform(0, 1, self.input_shape).astype(
                self.dtype)
            self.moment = np.random.uniform(-1, 1,
                                            self.input_shape).astype(self.dtype)

            self.mean_grad = np.random.uniform(-1, 1, self.input_shape).astype(
                self.dtype)
            self.mean_grad_out = np.random.uniform(
                -1, 1, self.input_shape).astype(self.dtype)

            param_out, mean_square_out, moment_out = calculate_rmsprop_by_numpy(
                param=self.param,
                grad=self.grad,
                mean_square=self.mean_square,
                moment=self.moment,
                learning_rate=self.learning_rate,
                epsilon=self.epsilon,
                decay=self.decay,
                momentum=self.momentum)
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
                'centered':
                False,  # TODO(houj04): when XDNN api supports 'center = True', add more test cases
            }
            self.outputs = {
                'ParamOut': param_out,
                'MomentOut': moment_out,
                'MeanSquareOut': mean_square_out,
                'MeanGradOut': self.mean_grad_out
            }

        def init_dtype(self):
            self.dtype = np.float32

        def test_check_output(self):
            self.check_output_with_place(self.place,
                                         no_check_set=['MeanGradOut'])

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


support_types = get_xpu_op_support_types('rmsprop')
for stype in support_types:
    create_test_class(globals(), XPUTestRMSPropOP, stype)

if __name__ == "__main__":
    unittest.main()
