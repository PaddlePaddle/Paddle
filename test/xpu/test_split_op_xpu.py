# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestSplitOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'split'
        self.use_dynamic_create_class = False

    # test with attr(num)
    class TestSplitOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.__class__.use_xpu = True
            self.__class__.op_type = 'split'
            self.use_mkldnn = False
            self.initParameters()
            self.inputs = {'X': self.x}
            self.attrs = {
                'axis': self.axis,
                'sections': self.sections,
                'num': self.num,
            }

            out = np.split(self.x, self.indices_or_sections, self.axis)
            self.outputs = {
                'Out': [(f'out{i}', out[i]) for i in range(len(out))]
            }

        def init_dtype(self):
            self.dtype = self.in_type

        def initParameters(self):
            self.x = np.random.random((4, 5, 6)).astype(self.dtype)
            self.axis = 2
            self.sections = []
            self.num = 3
            self.indices_or_sections = 3

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

    # unknown sections
    class TestSplitOp1(TestSplitOp):
        def initParameters(self):
            self.x = np.random.random((4, 5, 6)).astype(self.dtype)
            self.axis = 2
            self.sections = [2, 1, -1]
            self.num = 0
            self.indices_or_sections = [2, 3]

    # test with int32
    class TestSplitOp2(TestSplitOp):
        def initParameters(self):
            self.x = np.random.random((4, 5, 6)).astype(np.int32)
            self.axis = 2
            self.sections = []
            self.num = 3
            self.indices_or_sections = 3

    # test sections list contains 0
    class TestSplitOp3(TestSplitOp):
        def initParameters(self):
            self.x = np.random.random((4, 5, 6)).astype(self.dtype)
            self.axis = 1
            self.sections = [2, 0, 3]
            self.num = 3
            self.indices_or_sections = [2, 2]

    @check_run_big_shape_test()
    class TestSplitOpLargeShape1(TestSplitOp):
        def initParameters(self):
            self.x = np.random.random((1, 8192, 5, 384)).astype(self.dtype)
            self.sections = []
            self.axis = 3
            self.num = 3
            self.indices_or_sections = 3


support_types = get_xpu_op_support_types('split')
for stype in support_types:
    create_test_class(globals(), XPUTestSplitOp, stype)

if __name__ == '__main__':
    unittest.main()
