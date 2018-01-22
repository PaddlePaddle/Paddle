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

from op_test import OpTest


def layer_norm_naive(x, scale, beta, epsilon):
    n, c, h, w = x.shape
    mean = np.mean(x, axis=(1, 2, 3))
    var = np.var(x, axis=(1, 2, 3)) + epsilon
    output = scale * np.divide((x - mean.reshape([n, 1, 1, 1])),
                               (np.sqrt(var)).reshape([n, 1, 1, 1])) + beta
    return output, mean, var


class TestLayerNormdOp(OpTest):
    def setUp(self):
        self.init_test_case()

        input = np.random.random(self.input_size).astype("float32")
        self.inputs = {
            'X': input,
            'Scale': np.array([self.scale]).astype("float32"),
            'Bias': np.array([self.bias]).astype("float32")
        }
        output, mean, var = layer_norm_naive(input, self.scale, self.bias,
                                             self.epsilon)
        self.outputs = {'Y': output, 'Mean': mean, 'Variance': var}

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(
    #         ['Scale', 'Bias', 'X'], ['Y', 'Mean', 'Variance'],
    #         max_relative_error=0.02)

    def test_check_grad_no_x(self):
        self.check_grad(
            ['Scale', 'Bias'], ['Y', 'Mean', 'Variance'],
            max_relative_error=0.02,
            no_grad_set=set(['X']))

    # def test_check_grad_no_scale(self):
    #     self.check_grad(
    #         ['Bias','X'],
    #         'Y',
    #         max_relative_error=0.02,
    #         no_grad_set=set(['Scale']))
    #
    # def test_check_grad_no_bias(self):
    #     self.check_grad(
    #         ['Scale','X'],
    #         'Y',
    #         max_relative_error=0.02,
    #         no_grad_set=set(['Bias']))

    def init_test_case(self):
        self.op_type = "layer_norm"
        self.input_size = [2, 3, 4, 5]
        self.scale = 0.21
        self.bias = 0.1
        self.epsilon = 0.00001


if __name__ == '__main__':
    unittest.main()
