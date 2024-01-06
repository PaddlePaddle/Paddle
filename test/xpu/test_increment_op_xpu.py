#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


class XPUTestIncrementOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'increment'
        self.use_dynamic_create_class = False

    class TestXPUIncrementOp(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.op_type = 'increment'

            self.initTestCase()

            x = np.random.uniform(-100, 100, [1]).astype(self.dtype)
            output = x + np.asarray(self.step).astype(self.dtype)
            output = output.astype(self.dtype)

            self.inputs = {'X': x}
            self.attrs = {'step': self.step}
            self.outputs = {'Out': output}

        def initTestCase(self):
            self.step = -1.5

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

    class TestIncrement1(TestXPUIncrementOp):
        def initTestCase(self):
            self.step = 6.0

    class TestIncrement2(TestXPUIncrementOp):
        def initTestCase(self):
            self.step = 2.1

    class TestIncrement3(TestXPUIncrementOp):
        def initTestCase(self):
            self.step = -1.5

    class TestIncrement4(TestXPUIncrementOp):
        def initTestCase(self):
            self.step = 0.5

    class TestIncrement5(TestXPUIncrementOp):
        def initTestCase(self):
            self.step = 3


support_types = get_xpu_op_support_types('increment')
for stype in support_types:
    create_test_class(globals(), XPUTestIncrementOP, stype)

if __name__ == '__main__':
    unittest.main()
