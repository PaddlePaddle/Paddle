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
import paddle.fluid.core as core
from op_test import OpTest


class ElementwiseDivOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.dtype = np.float64
        self.init_dtype()
        """ Warning
        CPU gradient check error!
        'X': np.random.random((32,84)).astype("float32"),
        'Y': np.random.random((32,84)).astype("float32")
        """
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.05)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.05, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Y'))

    def init_dtype(self):
        pass


class TestElementwiseDivOp_scalar(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype(np.float64),
            'Y': np.random.uniform(0.1, 1, [1]).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] / self.inputs['Y']}


class TestElementwiseDivOp_Vector(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [32]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [32]).astype("float64")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_broadcast_0(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2]).astype("float64")
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(2, 1, 1))
        }


class TestElementwiseDivOp_broadcast_1(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [3]).astype("float64")
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 3, 1))
        }


class TestElementwiseDivOp_broadcast_2(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [4]).astype("float64")
        }

        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 1, 4))
        }


class TestElementwiseDivOp_broadcast_3(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [3, 4]).astype("float64")
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 3, 4, 1))
        }


class TestElementwiseDivOp_broadcast_4(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 1, 4]).astype("float64")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_broadcast_5(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 1, 5]).astype("float64")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_commonuse_1(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [1, 1, 4]).astype("float64"),
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_commonuse_2(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 1, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 1, 4, 1]).astype("float64"),
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_xsize_lessthan_ysize(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [4, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
        }

        self.attrs = {'axis': 2}

        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_INT(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.dtype = np.int32
        self.init_dtype()
        self.inputs = {
            'X': np.random.randint(
                1, 5, size=[2, 3]).astype(self.dtype),
            'Y': np.random.randint(
                1, 5, size=[2, 3]).astype(self.dtype)
        }
        self.outputs = {'Out': self.inputs['X'] // self.inputs['Y']}

    def test_check_output(self):
        self.check_output()

    def init_dtype(self):
        pass


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestElementwiseDivOpFp16(ElementwiseDivOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=1)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=1, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=1, no_grad_set=set('Y'))


if __name__ == '__main__':
    unittest.main()
