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
        self.config()
        self.set_case()
        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': self.result}

    def set_case(self):
        self.x = np.random.uniform(
            self.lbound, self.hbound, self.x_shape
        ).astype(self.dtype)
        self.y = np.random.uniform(
            self.lbound, self.hbound, self.y_shape
        ).astype(self.dtype)
        self.result = self.compute(self.x, self.y)

    def config(self):
        self.dtype = np.float32
        self.op_type = 'less_than'
        self.compute = np.less
        self.lbound = -100
        self.hbound = 100
        self.x_shape = [11, 17]
        self.y_shape = [11, 17]

    def test_check_output(self):
        paddle.enable_static()
        self.check_output_with_place(self.place)


class XPUTestLessThanOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'less_than'
        self.use_dynamic_create_class = False

    class LessThanOpTestCase1(TestCompareOpBase):
        def config(self):
            self.dtype = self.in_type
            self.op_type = 'less_than'
            self.compute = np.less
            self.set_data()

        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = [11, 17]

    class LessThanOpTestCase2(LessThanOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [11, 17]
            self.y_shape = [1]

    class LessThanOpTestCase3(LessThanOpTestCase1):
        def set_data(self):
            self.lbound = -300
            self.hbound = 300
            self.x_shape = [11, 17, 29]
            self.y_shape = [1]

    class LessThanOpTestCase4(LessThanOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [128, 128, 512]
            self.y_shape = [1]

    class LessThanOpTestCase5(LessThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [128, 128, 512]
            self.y_shape = [128, 128, 512]

    class LessThanOpTestCase_ZeroDim1(LessThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = []

    class LessThanOpTestCase_ZeroDim2(LessThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = [11, 17]

    class LessThanOpTestCase_ZeroDim3(LessThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = []


support_types = get_xpu_op_support_types('less_than')
for stype in support_types:
    create_test_class(globals(), XPUTestLessThanOP, stype)


class XPUTestLessEqualOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'less_equal'
        self.use_dynamic_create_class = False

    class LessEqualOpTestCase1(TestCompareOpBase):
        def config(self):
            self.dtype = self.in_type
            self.op_type = 'less_equal'
            self.compute = np.less_equal
            self.set_data()

        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = [11, 17]

    class LessEqualOpTestCase2(LessEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17, 255]
            self.y_shape = [11, 17, 255]

    class LessEqualOpTestCase3(LessEqualOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [11, 17, 255]
            self.y_shape = [1]

    class LessEqualOpTestCase4(LessEqualOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [11, 17]
            self.y_shape = [1]

    class LessEqualOpTestCase5(LessEqualOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [128, 128, 512]
            self.y_shape = [128, 128, 512]

    class LessEqualOpTestCase_ZeroDim1(LessEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = []

    class LessEqualOpTestCase_ZeroDim2(LessEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = [11, 17]

    class LessEqualOpTestCase_ZeroDim3(LessEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = []


support_types = get_xpu_op_support_types('less_equal')
for stype in support_types:
    create_test_class(globals(), XPUTestLessEqualOp, stype)


class XPUTestGreaterThanOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'greater_than'
        self.use_dynamic_create_class = False

    class GreaterThanOpTestCase1(TestCompareOpBase):
        def config(self):
            self.dtype = self.in_type
            self.op_type = 'greater_than'
            self.compute = np.greater
            self.set_data()

        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [128, 128, 512]
            self.y_shape = [128, 128, 512]

    class GreaterThanOpTestCase2(GreaterThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [128, 128, 512]
            self.y_shape = [1]

    class GreaterThanOpTestCase3(GreaterThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = [1]

    class GreaterThanOpTestCase4(GreaterThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = [11, 17]

    class GreaterThanOpTestCase5(GreaterThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [10, 10, 20, 20]
            self.y_shape = [10, 10, 20, 20]

    class GreaterThanOpTestCase_ZeroDim1(GreaterThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = []

    class GreaterThanOpTestCase_ZeroDim2(GreaterThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = [11, 17]

    class GreaterThanOpTestCase_ZeroDim3(GreaterThanOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = []


support_types = get_xpu_op_support_types('greater_than')
for stype in support_types:
    create_test_class(globals(), XPUTestGreaterThanOp, stype)


class XPUTestGreaterEqualOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'greater_equal'
        self.use_dynamic_create_class = False

    class GreaterEqualOpTestCase1(TestCompareOpBase):
        def config(self):
            self.dtype = self.in_type
            self.op_type = 'greater_equal'
            self.compute = np.greater_equal
            self.set_data()

        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [10, 10, 20, 20]
            self.y_shape = [10, 10, 20, 20]

    class GreaterEqualOpTestCase2(GreaterEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [10, 10]
            self.y_shape = [10, 10]

    class GreaterEqualOpTestCase3(GreaterEqualOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [512, 512, 2]
            self.y_shape = [1]

    class GreaterEqualOpTestCase4(GreaterEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [10, 10, 20, 20]
            self.y_shape = [1]

    class GreaterEqualOpTestCase5(GreaterEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [10, 30, 15]
            self.y_shape = [10, 30, 15]

    class GreaterEqualOpTestCase_ZeroDim1(GreaterEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = []

    class GreaterEqualOpTestCase_ZeroDim2(GreaterEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = [11, 17]

    class GreaterEqualOpTestCase_ZeroDim3(GreaterEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = []


support_types = get_xpu_op_support_types('greater_equal')
for stype in support_types:
    create_test_class(globals(), XPUTestGreaterEqualOp, stype)


class XPUTestEqualOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'equal'
        self.use_dynamic_create_class = False

    class EqualOpTestCase1(TestCompareOpBase):
        def config(self):
            self.dtype = self.in_type
            self.op_type = 'equal'
            self.compute = np.equal
            self.set_data()

        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [10, 30, 15]
            self.y_shape = [10, 30, 15]

    class EqualOpTestCase2(EqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [10, 30, 15]
            self.y_shape = [1]

    class EqualOpTestCase3(EqualOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [10, 30]
            self.y_shape = [10, 30]

    class EqualOpTestCase4(EqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [256, 256, 10]
            self.y_shape = [256, 256, 10]

    class EqualOpTestCase5(EqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = [1]

    class EqualOpTestCase_ZeroDim1(EqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = []

    class EqualOpTestCase_ZeroDim2(EqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = [11, 17]

    class EqualOpTestCase_ZeroDim3(EqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = []


support_types = get_xpu_op_support_types('equal')
for stype in support_types:
    create_test_class(globals(), XPUTestEqualOp, stype)


class XPUTestNotEqualOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'not_equal'
        self.use_dynamic_create_class = False

    class NotEqualOpTestCase1(TestCompareOpBase):
        def config(self):
            self.dtype = self.in_type
            self.op_type = 'not_equal'
            self.compute = np.not_equal
            self.set_data()

        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = [1]

    class NotEqualOpTestCase2(NotEqualOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [11, 17]
            self.y_shape = [11, 17]

    class NotEqualOpTestCase3(NotEqualOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [11, 17, 30]
            self.y_shape = [1]

    class NotEqualOpTestCase4(NotEqualOpTestCase1):
        def set_data(self):
            self.lbound = -200
            self.hbound = 200
            self.x_shape = [256, 256, 10]
            self.y_shape = [256, 256, 10]

    class NotEqualOpTestCase5(NotEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [512, 128]
            self.y_shape = [512, 128]

    class NotEqualOpTestCase_ZeroDim1(NotEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = []

    class NotEqualOpTestCase_ZeroDim2(NotEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = []
            self.y_shape = [11, 17]

    class NotEqualOpTestCase_ZeroDim3(NotEqualOpTestCase1):
        def set_data(self):
            self.lbound = -100
            self.hbound = 100
            self.x_shape = [11, 17]
            self.y_shape = []


support_types = get_xpu_op_support_types('not_equal')
for stype in support_types:
    create_test_class(globals(), XPUTestNotEqualOp, stype)

if __name__ == '__main__':
    unittest.main()
