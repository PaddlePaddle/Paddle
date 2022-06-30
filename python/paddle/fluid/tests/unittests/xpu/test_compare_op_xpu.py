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
        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': self.result}

    def set_case(self):
        self.op_type = 'less_than'
        self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
        self.y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
        self.result = np.less(self.x, self.y)

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

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.result = np.less(self.x, self.y)

    class LessThanOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'less_than'

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            self.result = np.less(self.x, self.y)


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

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.result = np.less_equal(self.x, self.y)

    class LessEqualOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'less_equal'

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            self.result = np.less_equal(self.x, self.y)


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

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.result = np.greater(self.x, self.y)

    class GreaterThanOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'greater_than'

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            self.result = np.greater(self.x, self.y)


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

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.result = np.greater_equal(self.x, self.y)

    class GreaterEqualOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'greater_equal'

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            self.result = np.greater_equal(self.x, self.y)


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

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.result = np.equal(self.x, self.y)

    class EqualOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'equal'

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            self.result = np.equal(self.x, self.y)


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

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.result = np.not_equal(self.x, self.y)

    class NotEqualOpTestCase2(TestCompareOpBase):

        def set_case(self):
            self.dtype = self.in_type
            self.op_type = 'not_equal'

            self.x = np.random.uniform(-100, 100, [11, 17]).astype(self.dtype)
            self.y = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            self.result = np.not_equal(self.x, self.y)


support_types = get_xpu_op_support_types('not_equal')
for stype in support_types:
    create_test_class(globals(), XPUTestNotEqualOp, stype)

if __name__ == '__main__':
    unittest.main()
