# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core

paddle.enable_static()
SEED = 2021


class TestFillConstant(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "fill_constant"
        self.init_dtype()

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'value': 3.8}
        self.outputs = {'Out': np.full((123, 92), 3.8)}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillConstantInt(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {
            'shape': [123, 92],
            'value': 1,
            'dtype': core.VarDesc.VarType.INT32
        }
        self.outputs = {'Out': np.full((123, 92), 1).astype(self.dtype)}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.int32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillConstantInt64(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {
            'shape': [123, 92],
            'value': 1,
            'dtype': core.VarDesc.VarType.INT64
        }
        self.outputs = {'Out': np.full((123, 92), 1).astype(self.dtype)}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.int64

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillConstantFP16(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {
            'shape': [123, 92],
            'value': 1.0,
            'dtype': core.VarDesc.VarType.FP16
        }
        self.outputs = {'Out': np.full((123, 92), 1.0).astype(self.dtype)}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestFillConstantBool(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {
            'shape': [123, 92],
            'value': True,
            'dtype': core.VarDesc.VarType.BOOL
        }
        self.outputs = {'Out': np.full((123, 92), True).astype(self.dtype)}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.BOOL

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillConstantWithPlaceType(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "fill_constant"
        self.init_dtype()

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'value': 3.8, 'place_type': 4}
        self.outputs = {'Out': np.full((123, 92), 3.8)}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == '__main__':
    unittest.main()
