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


class TestDpsgdOp(OpTest):

    def setUp(self):
        '''Test Dpsgd Operator with supplied attributes
        '''
        self.op_type = "dpsgd"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")

        learning_rate = 0.001
        clip = 10000.0
        batch_size = 16.0
        sigma = 0.0

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'LearningRate': np.array([learning_rate]).astype("float32")
        }

        self.attrs = {'clip': clip, 'batch_size': batch_size, 'sigma': sigma}

        param_out = dpsgd_step(self.inputs, self.attrs)

        self.outputs = {'ParamOut': param_out}

    def test_check_output(self):
        self.check_output(check_eager=True)


def dpsgd_step(inputs, attributes):
    '''
    Simulate one step of the dpsgd optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment, inf_norm and
    beta1 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    lr = inputs['LearningRate']

    clip = attributes['clip']
    batch_size = attributes['batch_size']
    sigma = attributes['sigma']

    param_out = param - lr * grad

    return param_out


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    unittest.main()
