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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle


class XPUTestXPUTransposeOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'transpose'
        self.use_dynamic_create_class = False

    class TestXPUTransposeOp(XPUOpTest):
        def setUp(self):
            self.init_op_type()
            self.init_type()
            self.initTestCase()
            self.use_xpu = True
            self.use_mkldnn = False
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            self.attrs = {
                'axis': list(self.axis),
                'use_mkldnn': False,
                'use_xpu': True,
            }
            self.outputs = {
                'XShape': np.random.random(self.shape).astype(self.dtype),
                'Out': self.inputs['X'].transpose(self.axis),
            }

        def init_op_type(self):
            self.op_type = "transpose2"
            self.use_mkldnn = False

        def init_type(self):
            self.dtype = self.in_type

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                place = paddle.XPUPlace(0)
                self.check_output_with_place(
                    place=place, no_check_set=['XShape']
                )

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X'], 'Out')

        def initTestCase(self):
            self.shape = (3, 40)
            self.axis = (1, 0)

    class TestCase_ZeroDim(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = ()
            self.axis = ()

    class TestCase0(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (100,)
            self.axis = (0,)

    class TestCase1(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (3, 4, 10)
            self.axis = (0, 2, 1)

    class TestCase2(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5)
            self.axis = (0, 2, 3, 1)

    class TestCase3(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6)
            self.axis = (4, 2, 3, 1, 0)

    class TestCase4(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5, 6, 1)
            self.axis = (4, 2, 3, 1, 0, 5)

    class TestCase5(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (2, 16, 96)
            self.axis = (0, 2, 1)

    class TestCase6(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (2, 10, 12, 16)
            self.axis = (3, 1, 2, 0)

    class TestCase7(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (2, 10, 2, 16)
            self.axis = (0, 1, 3, 2)

    class TestCase8(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
            self.axis = (0, 1, 3, 2, 4, 5, 6, 7)

    class TestCase9(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
            self.axis = (6, 1, 3, 5, 0, 2, 4, 7)

    class TestCase10(TestXPUTransposeOp):
        def initTestCase(self):
            self.shape = (200, 3, 2)
            self.axis = (-1, 1, -3)


support_types = get_xpu_op_support_types('transpose')
for stype in support_types:
    create_test_class(globals(), XPUTestXPUTransposeOp, stype)

if __name__ == "__main__":
    unittest.main()
