#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class ElementwiseMulOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'))


class TestElementwiseMulOp_scalar(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float32),
            'Y': np.random.rand(1).astype(np.float32)
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


class TestElementwiseMulOp_Vector(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.random((32, )).astype("float64"),
            'Y': np.random.random((32, )).astype("float64")
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMulOp_broadcast_0(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float64),
            'Y': np.random.rand(2).astype(np.float64)
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(2, 1, 1)
        }


class TestElementwiseMulOp_broadcast_1(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float64),
            'Y': np.random.rand(3).astype(np.float64)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 3, 1)
        }


class TestElementwiseMulOp_broadcast_2(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype(np.float64),
            'Y': np.random.rand(4).astype(np.float64)
        }

        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 1, 4)
        }


class TestElementwiseMulOp_broadcast_3(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 4, 5).astype(np.float64),
            'Y': np.random.rand(3, 4).astype(np.float64)
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 3, 4, 1)
        }


class TestElementWiseMulSelectedRows(OpTest):
    def setUp(self):
        self.rows = [0, 1, 2, 3, 4, 5, 6]
        self.feature = 12
        self.height = 100
        self.input_shape = (len(self.rows), self.feature)

    def prepare_input(self, scope, place):
        self.input = {
            "X": np.random.random(self.input_shape).astype("float32"),
            "Y": np.random.random(self.input_shape).astype("float32")
        }

        def init_input(in_name):
            x_selected_rows = scope.var(in_name).get_selected_rows()
            x_selected_rows.set_height(self.height)
            x_selected_rows.set_rows(self.rows)
            x_array = self.input[in_name]
            x_tensor = x_selected_rows.get_tensor()
            x_tensor.set(x_array, place)

        init_input("X")
        init_input("Y")

    def create_out_selected_row(self, scope):
        return scope.var('Out').get_selected_rows()

    def check_result(self, out_selected_rows):
        assert out_selected_rows.height() == self.height
        assert out_selected_rows.rows() == self.rows
        out_tensor = np.array(out_selected_rows.get_tensor())
        assert out_tensor.shape == self.input_shape

    def check_with_place(self, place):
        scope = core.Scope()
        self.prepare_input(scope, place)

        out_selected_rows = self.create_out_selected_row(scope)
        out_selected_rows.set_height(0)
        out_selected_rows.set_rows([])

        elementwise_mul = Operator("elementwise_mul", X='X', Y='Y', Out='Out')
        elementwise_mul.run(scope, place)
        self.check_result(out_selected_rows)

    def test_elewisemul_with_selected_rows_input(self):
        places = [core.CPUPlace()]
        for place in places:
            self.check_with_place(place)


if __name__ == '__main__':
    unittest.main()
