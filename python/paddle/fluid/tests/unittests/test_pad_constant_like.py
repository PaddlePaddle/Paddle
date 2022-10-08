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
from op_test import OpTest, check_out_dtype
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestPadConstantLikeOp(OpTest):

    def setUp(self):
        self.initTestCase()
        self.op_type = "pad_constant_like"
        self.inputs = {
            'X': np.random.random(self.x_shape).astype("float64"),
            'Y': np.random.random(self.y_shape).astype("float64")
        }
        self.attrs = {}
        self.attrs['pad_value'] = self.pad_value
        self.outputs = {
            'Out':
            np.pad(self.inputs['Y'],
                   self.paddings,
                   mode='constant',
                   constant_values=self.pad_value)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Y'], 'Out')

    def initTestCase(self):
        self.x_shape = (16, 40)
        self.y_shape = (3, 40)
        self.pad_value = 0.1
        self.paddings = [(0, 13), (0, 0)]


class TestCase1(TestPadConstantLikeOp):

    def initTestCase(self):
        self.x_shape = (4, 3, 4, 5)
        self.y_shape = (2, 3, 4, 5)
        self.paddings = [(0, 2), (0, 0), (0, 0), (0, 0)]
        self.pad_value = 0.5


class TestCase2(TestPadConstantLikeOp):

    def initTestCase(self):
        self.x_shape = (4, 3, 4, 10)
        self.y_shape = (2, 3, 2, 10)
        self.paddings = [(0, 2), (0, 0), (0, 2), (0, 0)]
        self.pad_value = 0.5


class TestPadConstantLikeOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            x_data = np.random.random((2, 2, 2, 2)).astype("float32")
            y_data = np.random.random((2, 2, 2, 2)).astype("float32")

            def test_Variable_x():
                var_y = fluid.data(name="data_y",
                                   shape=[2, 2, 2, 2],
                                   dtype="float32")
                fluid.layers.pad_constant_like(x=x_data, y=var_y)

            self.assertRaises(TypeError, test_Variable_x)

            def test_Variable_y():
                var_x = fluid.data(name="data_x",
                                   shape=[2, 2, 2, 2],
                                   dtype="float32")
                fluid.layers.pad_constant_like(x=var_x, y=y_data)

            self.assertRaises(TypeError, test_Variable_y)


class TestOutDtype(unittest.TestCase):

    def test_dtype(self):
        api_fn = fluid.layers.pad_constant_like
        check_out_dtype(api_fn,
                        in_specs=[([2, 3, 2, 3], 'float64'), ([1, 3, 1, 3], )],
                        expect_dtypes=['float32', 'float64', 'int32', 'int64'],
                        target_index=1,
                        pad_value=0.)


if __name__ == '__main__':
    unittest.main()
