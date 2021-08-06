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

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()


class TestFlattenOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "flatten_contiguous_range"
        self.place = paddle.NPUPlace(0)

        self.start_axis = 0
        self.stop_axis = -1
        self.dtype = np.float64
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.in_shape).astype(self.dtype)}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.in_shape).astype("float32")
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=["XShape"])

    def test_check_grad(self):
        pass
        #self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = -1
        self.new_shape = (120)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_1(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 1
        self.stop_axis = 2
        self.new_shape = (3, 10, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_2(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 1
        self.new_shape = (6, 5, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_3(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 2
        self.new_shape = (30, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_4(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = -2
        self.stop_axis = -1
        self.new_shape = (3, 2, 20)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_5(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 2
        self.stop_axis = 2
        self.new_shape = (3, 2, 5, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOpSixDims(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 3, 2, 4, 4)
        self.start_axis = 3
        self.stop_axis = 5
        self.new_shape = (3, 2, 3, 32)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_Float32(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 1
        self.new_shape = (6, 5, 4)
        self.dtype = np.float32

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_int(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 1
        self.new_shape = (6, 5, 4)
        self.dtype = np.int

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_uint8(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 1
        self.new_shape = (6, 5, 4)
        self.dtype = np.uint8

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_int8(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 1
        self.new_shape = (6, 5, 4)
        self.dtype = np.int8

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


class TestFlattenOp_int64(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 1
        self.new_shape = (6, 5, 4)
        self.dtype = np.int64

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis
        }


if __name__ == "__main__":
    unittest.main()
