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


def huber_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


class TestHuberLossOp(OpTest):
    def setUp(self):
        self.op_type = 'huber_loss'
        self.samples_num = 64
        self.delta = 1.0
        self.init_input()
        residual = self.inputs['Y'].reshape(
            self.samples_num, 1) - self.inputs['X'].reshape(self.samples_num, 1)
        loss = np.vectorize(huber_loss_forward)(residual,
                                                self.delta).astype('float32')
        self.attrs = {'delta': self.delta}
        self.outputs = {
            'Residual': residual,
            'Out': loss.reshape((self.samples_num, 1))
        }

    def init_input(self):
        self.inputs = {
            'X': np.random.uniform(0, 1.,
                                   (self.samples_num, 1)).astype('float32'),
            'Y': np.random.uniform(0, 1.,
                                   (self.samples_num, 1)).astype('float32'),
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.008)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.008, no_grad_set=set("residual"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.008, no_grad_set=set('residual'))


def TestHuberLossOp1(TestHuberLossOp):
    def init_input(self):
        self.inputs = {
            'X': np.random.uniform(0, 1.,
                                   (self.samples_num, 1)).astype('float32'),
            'Y': np.random.uniform(0, 1., (self.samples_num)).astype('float32'),
        }


if __name__ == '__main__':
    unittest.main()
