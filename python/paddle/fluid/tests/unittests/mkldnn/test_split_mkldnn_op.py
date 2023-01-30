#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle
=======
from __future__ import print_function
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.fluid.tests.unittests.op_test import OpTest


class TestSplitSectionsOneDNNOp(OpTest):
<<<<<<< HEAD
    def init_data_type(self):
        self.dtype = np.float32

    def init_x(self):
        if self.dtype == np.float32:
            self.x = np.random.random(self.input_shape).astype(self.dtype)
        elif self.dtype == np.int8:
            self.x = np.random.randint(-5, 5, self.input_shape).astype(
                self.dtype
            )
        else:  # uint8
            self.x = np.random.randint(0, 10, self.input_shape).astype(
                self.dtype
            )

    def init_test_case(self):
        self.input_shape = (4, 5, 6)
        self.init_x()
        self.axis = 1
        self.num = 0
        self.sections = [2, 1, 2]
=======

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("float32")
        self.axis = 1
        self.sections = [2, 1, 2]
        indices_or_sections = [2, 3]  # sections
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np_sections = [2, 3]
        self.out = np.split(self.x, np_sections, self.axis)

    def setUp(self):
        self.op_type = "split"
        self.axis_tensor = None
        self.sections_tensor_list = None
<<<<<<< HEAD
        self.init_data_type()
        self.init_test_case()
=======
        self.num = 0
        self.init_data()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': self.x}
        self.attrs = {'use_mkldnn': True, 'num': self.num}

        if self.axis is not None:
            self.attrs['axis'] = self.axis
        if self.sections is not None:
            self.attrs['sections'] = self.sections
        if self.axis_tensor is not None:
            self.inputs['AxisTensor'] = self.axis_tensor
        if self.sections_tensor_list is not None:
            self.inputs['SectionsTensorList'] = self.sections_tensor_list

<<<<<<< HEAD
        self.outputs = {
            'Out': [('out%d' % i, self.out[i]) for i in range(len(self.out))]
        }
=======
        self.outputs = {'Out': [('out%d' % i, self.out[i]) \
            for i in range(len(self.out))]}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


# test with attr(num)
class TestSplitNumOneDNNOp(TestSplitSectionsOneDNNOp):
<<<<<<< HEAD
    def init_test_case(self):
        self.input_shape = (4, 8, 5, 3)
        self.init_x()
        self.axis = 1
        self.num = 4
        self.sections = []
        indices_or_sections = 4  # indices
=======

    def init_data(self):
        self.x = np.random.random((4, 8, 5, 3)).astype("float32")
        self.axis = 1
        self.sections = []
        self.num = 4
        indices_or_sections = 4  #indices
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out = np.split(self.x, indices_or_sections, self.axis)

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2', 'out3'])


class TestSplitNumAxisTensorOneDNNOp(TestSplitSectionsOneDNNOp):
<<<<<<< HEAD
    def init_test_case(self):
        self.input_shape = (4, 5, 6)
        self.init_x()
        self.num = 3
        self.axis = None
        self.sections = []
        self.axis_tensor = np.array([2]).astype("int32")
        indices_or_sections = 3  # indices
=======

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("float32")
        self.axis = None
        self.sections = []
        self.num = 3
        indices_or_sections = 3  #indices
        self.axis_tensor = np.array([2]).astype("int32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out = np.split(self.x, indices_or_sections, 2)


# attr(sections) is list containing Tensor
class TestSplitSectionsTensorOneDNNOp(TestSplitSectionsOneDNNOp):
<<<<<<< HEAD
    def init_test_case(self):
        self.input_shape = (4, 5, 6)
        self.init_x()
        self.num = 0
=======

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.axis = 1
        self.sections = [2, 1, 2]
        self.sections_tensor_list = []
        for index, ele in enumerate(self.sections):
<<<<<<< HEAD
            self.sections_tensor_list.append(
                ("x" + str(index), np.ones((1)).astype('int32') * ele)
            )
        self.sections = [-1, -1, -1]
        indices_or_sections = [2, 3]  # sections
=======
            self.sections_tensor_list.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))
        self.sections = [-1, -1, -1]
        indices_or_sections = [2, 3]  #sections
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out = np.split(self.x, indices_or_sections, self.axis)


class TestSplitOpUnknownSectionOneDNNOp(TestSplitSectionsOneDNNOp):
<<<<<<< HEAD
    def init_test_case(self):
        self.input_shape = (4, 5, 6)
        self.init_x()
        self.num = 0
        self.axis = 2
        self.sections = [2, 2, -1]
        indices_or_sections = [2, 4]  # sections
        self.out = np.split(self.x, indices_or_sections, self.axis)


def create_test_class(parent):
    '''
    Create int8 and uint8 versions for each test. Parent tests work by default on fp32.
    '''

    class TestInt8Case(parent):
        def init_data_type(self):
            self.dtype = np.int8

        def test_check_grad(self):
            pass

    class TestUint8Case(parent):
        def init_data_type(self):
            self.dtype = np.uint8

        def test_check_grad(self):
            pass

    TestInt8Case.__name__ = "{0}_{1}".format(parent.__name__, "INT8")
    TestUint8Case.__name__ = "{0}_{1}".format(parent.__name__, "UINT8")
    globals()[TestInt8Case.__name__] = TestUint8Case
    globals()[TestUint8Case.__name__] = TestInt8Case


create_test_class(TestSplitNumOneDNNOp)
create_test_class(TestSplitNumAxisTensorOneDNNOp)
create_test_class(TestSplitSectionsTensorOneDNNOp)
create_test_class(TestSplitOpUnknownSectionOneDNNOp)
create_test_class(TestSplitSectionsOneDNNOp)

=======

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("float32")
        self.axis = 2
        self.sections = [2, 2, -1]
        indices_or_sections = [2, 4]  #sections
        self.out = np.split(self.x, indices_or_sections, self.axis)


>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
