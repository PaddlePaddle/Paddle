#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.v2.fluid.core as core
from op_test import OpTest


class TestMomentumOp1(OpTest):
    def setUp(self):
        self.op_type = "momentum"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        mu = 0.0001
        use_nesterov = False

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu}

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate + \
                        velocity_out * mu * learning_rate
        else:
            param_out = param - learning_rate * velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def test_check_output(self):
        self.check_output()


class TestMomentumOp2(OpTest):
    '''Test Momentum with default values for attributes
    '''

    def setUp(self):
        self.op_type = "momentum"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        mu = 0.0001
        use_nesterov = True

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu, 'use_nesterov': use_nesterov}

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate + \
                        velocity_out * mu * learning_rate
        else:
            param_out = param - learning_rate * velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def test_check_output(self):
        self.check_output()


class TestMomentumOp3(OpTest):
    '''Test Momentum with LARS attribute on
    '''

    def setUp(self):
        self.op_type = "momentum"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([2.56]).astype("float32")
        mu = 0.0001
        use_nesterov = False
        use_local_lr = True
        local_gw_ratio = 0.1
        weight_decay = 0.5

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {
            'mu': mu,
            'use_nesterov': use_nesterov,
            'use_local_lr': use_local_lr,
            'local_gw_ratio': local_gw_ratio,
            'weight_decay': weight_decay,
        }

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate + \
                        velocity_out * mu * learning_rate
        else:
            local_lr = learning_rate
            if use_local_lr:
                param_norm = np.sqrt(np.sum(np.square(param)))
                grad_norm = np.sqrt(np.sum(np.square(grad)))
                local_lr = learning_rate * local_gw_ratio * param_norm / \
                               (grad_norm + weight_decay * param_norm)
            param_out = param - local_lr * velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def test_check_output(self):
        place = core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
