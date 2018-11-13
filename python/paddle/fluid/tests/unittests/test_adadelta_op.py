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
from op_test import OpTest


class TestAdadeltaOp1(OpTest):
    def setUp(self):
        self.op_type = "adadelta"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The squared gradient is positive
        avg_squared_grad = np.random.random((102, 105)).astype("float32")
        # The squared update is positive
        avg_squared_update = np.random.random((102, 105)).astype("float32")

        rho = 0.95
        epsilon = 1e-6

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'AvgSquaredGrad': avg_squared_grad,
            'AvgSquaredUpdate': avg_squared_update
        }

        self.attrs = {'rho': rho, 'epsilon': epsilon}

        avg_squared_grad_out = rho * avg_squared_grad + \
            (1 - rho) * np.square(grad)
        update = -np.multiply(
            np.sqrt(
                np.divide(avg_squared_update + epsilon, avg_squared_grad_out +
                          epsilon)), grad)

        avg_squared_update_out = rho * avg_squared_update + \
            (1 - rho) * np.square(update)

        param_out = param + update

        self.outputs = {
            'ParamOut': param_out,
            'AvgSquaredGradOut': avg_squared_grad_out,
            'AvgSquaredUpdateOut': avg_squared_update_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdadeltaOp2(OpTest):
    '''Test Adadelta op with default attribute values
    '''

    def setUp(self):
        self.op_type = "adadelta"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The squared gradient is positive
        avg_squared_grad = np.random.random((102, 105)).astype("float32")
        # The squared update is positive
        avg_squared_update = np.random.random((102, 105)).astype("float32")

        rho = 0.95
        epsilon = 1e-6

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'AvgSquaredGrad': avg_squared_grad,
            'AvgSquaredUpdate': avg_squared_update
        }

        avg_squared_grad_out = rho * avg_squared_grad + \
            (1 - rho) * np.square(grad)
        update = -np.multiply(
            np.sqrt(
                np.divide(avg_squared_update + epsilon, avg_squared_grad_out +
                          epsilon)), grad)

        avg_squared_update_out = rho * avg_squared_update + \
            (1 - rho) * np.square(update)

        param_out = param + update

        self.outputs = {
            'ParamOut': param_out,
            'AvgSquaredGradOut': avg_squared_grad_out,
            'AvgSquaredUpdateOut': avg_squared_update_out
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
