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
import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


# test with attr(num)
class TestSplitOp(XPUOpTest):
    def initDefaultParameters(self):
        self.dtype = 'float32'
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def setUp(self):
        self.__class__.op_type = 'split'
        self.use_xpu = True
        self.use_mkldnn = False
        self.initDefaultParameters()
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
            'sections': self.sections,
            'num': self.num
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)


# unknown sections
class TestSplitOp_2(XPUOpTest):
    def initDefaultParameters(self):
        self.dtype = 'float32'
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = [2, 1, -1]
        self.num = 0
        self.indices_or_sections = [2, 3]

    def setUp(self):
        self.__class__.op_type = 'split'
        self.use_xpu = True
        self.use_mkldnn = False
        self.initDefaultParameters()
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
            'sections': self.sections,
            'num': self.num
        }
        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)


# test with int32
class TestSplitOp_5(XPUOpTest):
    def initDefaultParameters(self):
        self.dtype = 'int32'
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def setUp(self):
        self.__class__.op_type = 'split'
        self.use_xpu = True
        self.use_mkldnn = False
        self.initDefaultParameters()
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
            'sections': self.sections,
            'num': self.num
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)


if __name__ == '__main__':
    unittest.main()
