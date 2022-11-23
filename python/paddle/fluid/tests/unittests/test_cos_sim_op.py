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
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestCosSimOp(OpTest):

    def setUp(self):
        self.op_type = "cos_sim"
        self.inputs = {
            'X': np.random.random((6, 20)).astype("float32"),
            'Y': np.random.random((6, 20)).astype("float32")
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
        self.check_grad(['Y'],
                        'Out',
                        max_relative_error=0.06,
                        no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'],
                        'Out',
                        max_relative_error=0.06,
                        no_grad_set=set('Y'))


class TestCosSimOp2(TestCosSimOp):

    def setUp(self):
        self.op_type = "cos_sim"
        self.inputs = {
            'X': np.random.random((6, 100)).astype("float32"),
            'Y': np.random.random((1, 100)).astype("float32")
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
            'X': np.random.random((6, 5, 4)).astype("float32"),
            'Y': np.random.random((6, 5, 4)).astype("float32")
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
            'X': np.random.random((6, 5, 20)).astype("float32"),
            'Y': np.random.random((1, 5, 20)).astype("float32")
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


class TestCosSimOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input of batch_norm must be Variable.
            x1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                         [[1, 1, 1, 1]], fluid.CPUPlace())
            x2 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                         [[1, 1, 1, 1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.cos_sim, x1, x2)

            # the input dtype of batch_norm must be float32
            x3 = fluid.layers.data(name='x3', shape=[3, 4, 5, 6], dtype="int32")
            x4 = fluid.layers.data(name='x4', shape=[3, 4, 5, 6], dtype="int64")
            self.assertRaises(TypeError, fluid.layers.cos_sim, x3, x4)


if __name__ == '__main__':
    unittest.main()
