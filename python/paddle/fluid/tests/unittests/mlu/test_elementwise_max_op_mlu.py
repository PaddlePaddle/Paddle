#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid.core import ops

paddle.enable_static()
SEED = 2022


class TestElementwiseMax(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.maximum(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseMaxFp16(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        y = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        out = np.maximum(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.device.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseMaxInt32(OpTest):

    def init_dtype(self):
        self.dtype = np.int32


class TestTestElementwiseMax_Vector(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [100]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float32")
        }
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseMax_broadcast_0(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [100, 3, 4]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float32")
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': np.maximum(self.inputs['X'],
                              self.inputs['Y'].reshape(100, 1, 1))
        }


class TestTestElementwiseMax_broadcast_1(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 100, 4]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float32")
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.maximum(self.inputs['X'],
                              self.inputs['Y'].reshape(1, 100, 1))
        }


class TestTestElementwiseMax_broadcast_2(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 100]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float32")
        }

        self.outputs = {
            'Out': np.maximum(self.inputs['X'],
                              self.inputs['Y'].reshape(1, 1, 100))
        }


class TestTestElementwiseMax_broadcast_3(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 10, 12, 5]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [10, 12]).astype("float32")
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.maximum(self.inputs['X'], self.inputs['Y'].reshape(1, 10, 12, 1))
        }


class TestTestElementwiseMax_broadcast_4(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 50]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [2, 1, 50]).astype("float32")
        }
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseMax_broadcast_5(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 20]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 1, 20]).astype("float32")
        }
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseMax_commonuse_1(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 100]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [1, 1, 100]).astype("float32"),
        }
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseMax_commonuse_2(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [30, 3, 1, 5]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [30, 1, 4, 1]).astype("float32"),
        }
        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseMax_xsize_lessthan_ysize(TestElementwiseMax):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_max"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [10, 12]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 10, 12]).astype("float32"),
        }

        self.attrs = {'axis': 2}

        self.outputs = {'Out': np.maximum(self.inputs['X'], self.inputs['Y'])}


if __name__ == '__main__':
    unittest.main()
