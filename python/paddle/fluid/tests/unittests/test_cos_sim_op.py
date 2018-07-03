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


class TestCosSimOp(OpTest):
    def setUp(self):
        self.op_type = "cos_sim"
        self.inputs = {
            'X': np.random.random((6, 5)).astype("float32"),
            'Y': np.random.random((6, 5)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=1)
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=1)
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=1) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.06)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.06, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.06, no_grad_set=set('Y'))


class TestCosSimOp2(TestCosSimOp):
    def setUp(self):
        self.op_type = "cos_sim"
        self.inputs = {
            'X': np.random.random((6, 5)).astype("float32"),
            'Y': np.random.random((1, 5)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=1)
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=1)
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=1) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }


class TestCosSimOp3(TestCosSimOp):
    def setUp(self):
        self.op_type = "cos_sim"
        self.inputs = {
            'X': np.random.random((6, 5, 2)).astype("float32"),
            'Y': np.random.random((6, 5, 2)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=(1, 2))
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=(1, 2))
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=(1, 2)) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }


class TestCosSimOp4(TestCosSimOp):
    def setUp(self):
        self.op_type = "cos_sim"
        self.inputs = {
            'X': np.random.random((6, 5, 2)).astype("float32"),
            'Y': np.random.random((1, 5, 2)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=(1, 2))
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=(1, 2))
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=(1, 2)) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }


if __name__ == '__main__':
    unittest.main()
