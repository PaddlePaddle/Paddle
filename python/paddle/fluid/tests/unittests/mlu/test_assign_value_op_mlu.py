#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy
import sys

sys.path.append("..")

import op_test
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers

paddle.enable_static()
numpy.random.seed(2022)


class TestAssignValueMLUOp(op_test.OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "assign_value"
        self.inputs = {}
        self.attrs = {}
        self.init_data()

        self.attrs["shape"] = self.value.shape
        self.attrs["dtype"] = framework.convert_np_dtype_to_dtype_(
            self.value.dtype)
        self.outputs = {"Out": self.value}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def init_data(self):
        self.value = numpy.random.random(size=(2, 5)).astype(numpy.float32)
        self.attrs["fp32_values"] = [float(v) for v in self.value.flat]

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAssignValueMLUOp2(TestAssignValueMLUOp):

    def init_data(self):
        self.value = numpy.random.random(size=(2, 5)).astype(numpy.int32)
        self.attrs["int32_values"] = [int(v) for v in self.value.flat]


class TestAssignValueMLUOp3(TestAssignValueMLUOp):

    def init_data(self):
        self.value = numpy.random.random(size=(2, 5)).astype(numpy.int64)
        self.attrs["int64_values"] = [int(v) for v in self.value.flat]


class TestAssignValueMLUOp4(TestAssignValueMLUOp):

    def init_data(self):
        self.value = numpy.random.choice(a=[False, True],
                                         size=(2, 5)).astype(numpy.bool)
        self.attrs["bool_values"] = [int(v) for v in self.value.flat]


if __name__ == '__main__':
    unittest.main()
