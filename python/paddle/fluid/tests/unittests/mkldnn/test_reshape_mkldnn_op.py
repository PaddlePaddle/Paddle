#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core

from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_reshape_op import TestReshapeOp, TestReshapeOpWithInputShape


class TestMKLDNNReshape(TestReshapeOp):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.use_mkldnn = True
        self.initInput()
        self.inputs = {"X": self.input_data}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def test_check_grad(self):
        pass

    def initInput(self):
        self.input_data = np.random.random(self.ori_shape).astype('float32')
        self.data_type = 'float32'

    def init_data(self):
        self.ori_shape = (2, 2)
        self.new_shape = (1, 4)
        self.infered_shape = (1, 4)

    def test_check_output(self):
        self.check_output_with_place(
            core.CPUPlace(), atol=0, no_check_set=['XShape'])


class TestMKLDNNReshapeOpDimInfer1(TestMKLDNNReshape):
    def init_data(self):
        self.ori_shape = (5, 10)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestMKLDNNReshapeOpDimInfer2(TestMKLDNNReshape):
    def init_data(self):
        self.ori_shape = (2, 2, 6)
        self.new_shape = (2, 0, 3, -1)
        self.infered_shape = (2, 2, 3, -1)


class TestMKLDNNReshapeOpWithInputShape(TestReshapeOpWithInputShape):
    def setUp(self):
        self.ori_shape = (6, 5)
        self.new_shape = (0, -1, 5)
        self.actual_shape = (2, 3, 5)
        self.use_mkldnn = True
        self.initInput()
        self.op_type = "reshape2"
        self.inputs = {
            "X": self.input_data,
            "Shape": np.array(
                self.actual_shape, dtype="int32")
        }
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.actual_shape),
            'XShape': np.random.random(self.ori_shape).astype(self.data_type)
        }

    def test_check_grad(self):
        pass

    def initInput(self):
        self.input_data = np.random.random(self.ori_shape).astype('float32')
        self.data_type = 'float32'


def create_test_class(parent):

    #--------------------test mkldnn reshape with fp32 input--------------------

    class TestFp32Case(parent):
        def initInput(self):
            self.input_data = np.random.random(self.ori_shape).astype('float32')
            self.data_type = 'float32'

#--------------------test mkldnn reshape with s8 input----------------------

    class TestINT8Case(parent):
        def initInput(self):
            self.input_data = (
                np.random.randint(0, 100, self.ori_shape) - 50).astype('int8')
            self.data_type = 'int8'

#--------------------test mkldnn reshape with u8 input----------------------

    class TestUINT8Case(parent):
        def initInput(self):
            self.input_data = (np.random.randint(
                0, 100, self.ori_shape)).astype('uint8')
            self.data_type = 'uint8'

    cls_name_fp32 = "{0}".format(parent.__name__)
    cls_name_int8 = "{0}".format(parent.__name__)
    cls_name_uint8 = "{0}".format(parent.__name__)
    TestFp32Case.__name__ = cls_name_fp32
    TestINT8Case.__name__ = cls_name_int8
    TestUINT8Case.__name__ = cls_name_uint8
    globals()[cls_name_fp32] = TestFp32Case
    globals()[cls_name_int8] = TestINT8Case
    globals()[cls_name_uint8] = TestUINT8Case

create_test_class(TestMKLDNNReshapeOpDimInfer1)
create_test_class(TestMKLDNNReshapeOpDimInfer2)
create_test_class(TestReshapeOpWithInputShape)

if __name__ == "__main__":
    unittest.main()
