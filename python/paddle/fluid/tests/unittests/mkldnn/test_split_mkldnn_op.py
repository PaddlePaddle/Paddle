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

from __future__ import print_function
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core
from paddle.fluid.tests.unittests.op_test import OpTest


class TestSplitSectionsOneDNNOp(OpTest):
    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("float32")
        self.axis = 1
        self.sections = [2, 1, 2]
        self.num = None
        indices_or_sections = [2, 3] # sections
        np_sections = [2, 3]
        self.out = np.split(self.x, np_sections, self.axis)

    def setUp(self):
        self.op_type = "split"
        self.axis_tensor = None
        self.init_data()
        self.inputs = {'X': self.x}
        self.attrs = {'use_mkldnn' : True}

        if self.axis is not None:
            self.attrs['axis'] = self.axis
        if self.num is not None:
            self.attrs['num'] = self.num
        if self.sections is not None:
            self.attrs['sections'] = self.sections
        if self.axis_tensor is not None:
            self.inputs['AxisTensor'] = self.axis_tensor

        self.outputs = {'Out': [('out%d' % i, self.out[i]) \
            for i in range(len(self.out))]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


# test with attr(num)
class TestSplitNumOneDNNop(TestSplitSectionsOneDNNOp):
    def init_data(self):
        self.x = np.random.random((4, 8, 5)).astype("float32")
        self.axis = 1
        self.sections = []
        self.num = 4
        indices_or_sections = 4 #indices
        self.out = np.split(self.x, indices_or_sections, self.axis)

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2', 'out3'])


class TestSplitNumAxisTensorOneDNN(TestSplitSectionsOneDNNOp):
    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("float32")
        self.axis = 1
        self.sections = []
        self.num = 3
        indices_or_sections = 3 #indices
        self.axis_tensor = np.array([2]).astype("int32")
        self.out = np.split(self.x, indices_or_sections, 2)

    def test_check_output(self):
        self.check_output()

#
# attr(sections) is list containing Tensor
#class TestSplitOp_SectionsTensor(OpTest):
#    def setUp(self):
#        self._set_op_type()
#        self.dtype = self.get_dtype()
#        self.init_data()
#        self.inputs = {'X': self.x}
#
#        sections_tensor = []
#        for index, ele in enumerate(self.sections):
#            sections_tensor.append(("x" + str(index), np.ones(
#                (1)).astype('int32') * ele))
#
#        self.inputs['SectionsTensorList'] = sections_tensor
#
#        self.attrs = {
#            'axis': self.axis,
#            'sections': self.sections_infer,
#            'num': self.num
#        }
#
#        out = np.split(self.x, self.indices_or_sections, self.axis)
#        self.outputs = {'Out': [('out%d' % i, out[i]) \
#                                for i in range(len(out))]}
#
#    def init_data(self):
#        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
#        self.axis = 1
#        self.sections = [2, 1, 2]
#        self.sections_infer = [-1, -1, -1]
#        self.num = 0
#        self.indices_or_sections = [2, 3]
#
#    def get_dtype(self):
#        return "float64"
#
#    def _set_op_type(self):
#        self.op_type = "split"
#
#    def test_check_output(self):
#        self.check_output()
#
#    def test_check_grad(self):
#        self.check_grad(['X'], ['out0', 'out1', 'out2'])


#class TestSplitOp_unk_section(OpTest):
#    def setUp(self):
#        self._set_op_type()
#        self.dtype = self.get_dtype()
#        self.init_data()
#        self.inputs = {'X': self.x}
#        self.attrs = {
#            'axis': self.axis,
#            'sections': self.sections,
#            'num': self.num
#        }
#
#        out = np.split(self.x, self.indices_or_sections, self.axis)
#        self.outputs = {'Out': [('out%d' % i, out[i]) \
#                                for i in range(len(out))]}
#
#    def init_data(self):
#        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
#        self.axis = 2
#        self.sections = [2, 1, -1]
#        self.num = 0
#        self.indices_or_sections = [2, 3]
#
#    def get_dtype(self):
#        return "float64"
#
#    def _set_op_type(self):
#        self.op_type = "split"
#
#    def test_check_output(self):
#        self.check_output()
#
#    def test_check_grad(self):
#        self.check_grad(['X'], ['out0', 'out1', 'out2'])

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
