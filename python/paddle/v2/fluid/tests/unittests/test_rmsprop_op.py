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
from op_test import OpTest


class TestRmspropOp1(OpTest):
    ''' Test RMSProp with explicit inputs
    '''

    def setUp(self):
        self.op_type = "rmsprop"

        param = np.random.random((123, 321)).astype("float32")
        mean_square = np.random.random((123, 321)).astype("float32")
        learning_rate = np.array([0.01]).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")

        epsilon = 1e-6
        decay = 0.9
        momentum = 0.0

        self.inputs = {
            'Param': param,
            'MeanSquare': mean_square,
            'LearningRate': learning_rate,
            'Grad': grad,
            'Moment': moment,
        }

        self.attrs = {'epsilon': epsilon, 'decay': decay, 'momentum': momentum}

        ms_out = decay * mean_square + (1 - decay) * grad * grad
        moment_out = momentum * moment + \
            learning_rate * grad / np.sqrt(ms_out + epsilon)
        param_out = param - moment_out

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'MeanSquareOut': ms_out
        }

    def test_check_output(self):
        self.check_output()


class TestRmspropOp2(OpTest):
    '''Test RMSProp with default values for attributes
    '''

    def setUp(self):
        self.op_type = "rmsprop"

        param = np.random.random((123, 321)).astype("float32")
        mean_square = np.random.random((123, 321)).astype("float32")
        learning_rate = np.array([0.01]).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")

        epsilon = 1.0e-10
        decay = 0.9
        momentum = 0.0

        self.inputs = {
            'Param': param,
            'MeanSquare': mean_square,
            'LearningRate': learning_rate,
            'Grad': grad,
            'Moment': moment,
        }

        ms_out = decay * mean_square + (1 - decay) * grad * grad
        moment_out = momentum * moment + \
            learning_rate * grad / np.sqrt(ms_out + epsilon)
        param_out = param - moment_out

        self.outputs = {
            'ParamOut': param_out,
            'MomentOut': moment_out,
            'MeanSquareOut': ms_out
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
