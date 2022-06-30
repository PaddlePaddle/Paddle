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

import sys

sys.path.append("..")
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types
from xpu.get_test_cover_info import XPUOpTestWrapper


class TestCompareOpBase(XPUOpTest):

    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.init_dtype()
        self.set_case()

    def set_case(self):
        self.op_type = 'less_than'
        input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
        input_y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
        result = np.less(input_x, input_y)
        self.inputs = {'X': input_x, 'Y': input_y}
        self.outputs = {'Out': result}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):

        paddle.enable_static()
        self.check_output_with_place(self.place)


class XPUTestLessThanOP(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'less_than'
        self.use_dynamic_create_class = False

    class LessThanOpTestCase1(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'less_than'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            result = np.less(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}

    class LessThanOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'less_than'
            # np.random.rand
            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            result = np.less(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}


support_types = get_xpu_op_support_types('less_than')
for stype in support_types:
    create_test_class(globals(), XPUTestLessThanOP, stype)


class XPUTestLessEqualOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'less_equal'
        self.use_dynamic_create_class = False

    class LessEqualOpTestCase1(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'less_equal'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            result = np.less_equal(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}

    class LessEqualOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'less_equal'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            result = np.less_equal(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}


support_types = get_xpu_op_support_types('less_equal')
for stype in support_types:
    create_test_class(globals(), XPUTestLessEqualOp, stype)


class XPUTestGreaterThanOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'greater_than'
        self.use_dynamic_create_class = False

    class GreaterThanOpTestCase1(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'greater_than'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            result = np.greater(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}

    class GreaterThanOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'greater_than'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            result = np.greater(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}


support_types = get_xpu_op_support_types('greater_than')
for stype in support_types:
    create_test_class(globals(), XPUTestGreaterThanOp, stype)


class XPUTestGreaterEqualOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'greater_equal'
        self.use_dynamic_create_class = False

    class GreaterEqualOpTestCase1(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'greater_equal'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            result = np.greater_equal(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}

    class GreaterEqualOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'greater_equal'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            result = np.greater_equal(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}


support_types = get_xpu_op_support_types('greater_equal')
for stype in support_types:
    create_test_class(globals(), XPUTestGreaterEqualOp, stype)


class XPUTestEqualOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'equal'
        self.use_dynamic_create_class = False

    class EqualOpTestCase1(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'equal'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            result = np.equal(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}

    class EqualOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'equal'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            result = np.equal(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}


support_types = get_xpu_op_support_types('equal')
for stype in support_types:
    create_test_class(globals(), XPUTestEqualOp, stype)


class XPUTestNotEqualOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'not_equal'
        self.use_dynamic_create_class = False

    class NotEqualOpTestCase1(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'not_equal'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            result = np.not_equal(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}

    class NotEqualOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'not_equal'

            input_x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            input_y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            result = np.not_equal(input_x, input_y)
            self.inputs = {'X': input_x, 'Y': input_y}
            self.outputs = {'Out': result}


support_types = get_xpu_op_support_types('not_equal')
for stype in support_types:
    create_test_class(globals(), XPUTestNotEqualOp, stype)

if __name__ == '__main__':
    unittest.main()
