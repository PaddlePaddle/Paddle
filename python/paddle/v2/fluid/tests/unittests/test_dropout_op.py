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


class TestDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('float32')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.05)


class TestDropoutOp2(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 1.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': np.zeros((32, 64)).astype('float32'),
            'Mask': np.zeros((32, 64)).astype('float32')
        }


class TestDropoutOp3(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 2)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('float32')
        }


class TestDropoutOp4(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.35, 'fix_seed': True, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }

    def test_check_output(self):
        self.check_output()


class TestDropoutOp5(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {'dropout_prob': 0.75, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
