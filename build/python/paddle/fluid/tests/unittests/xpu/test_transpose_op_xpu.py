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
import sys

sys.path.append("..")
import paddle

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
paddle.enable_static()


class XPUTestTransposeOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "transpose2"
        self.use_dynamic_create_class = False

    class TestTransposeOp(XPUOpTest):
        def setUp(self):
            self.init_op_type()
            self.set_case()
            self.init_data()
            self.use_xpu = True
            self.use_mkldnn = False

        def init_data(self):
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            self.attrs = {
                'axis': list(self.axis),
                'use_mkldnn': False,
                'use_xpu': True
            }
            self.outputs = {
                'XShape': np.random.random(self.shape).astype(self.dtype),
                'Out': self.inputs['X'].transpose(self.axis)
            }

        def init_op_type(self):
            self.op_type = "transpose2"
            self.use_mkldnn = False

        #it is important of transpose that shape and axis of input or output data
        def set_case(self):
            self.shape = (3, 40)
            self.axis = (1, 0)

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(
                    place=place, no_check_set=['XShape'])

        # get the grad from transpose
        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X'], 'Out')

    class TestXPUTransposeCase0(TestTransposeOp):
        def set_case(self):
            self.shape = (100, )
            self.axis = (0, )

    class TestXPUTransposeCase1(TestTransposeOp):
        def set_case(self):
            self.shape = (3, 4, 10)
            self.axis = (0, 2, 1)

    class TestXPUTransposeCase2(TestTransposeOp):
        def set_case(self):
            self.shape = (2, 3, 4, 5)
            self.axis = (0, 2, 3, 1)

    class TestXPUTransposeCase3(TestTransposeOp):
        def set_case(self):
            self.shape = (2, 3, 4, 5, 6)
            self.axis = (4, 2, 3, 1, 0)

    class TestXPUTransposeCase4(TestTransposeOp):
        def set_case(self):
            self.shape = (2, 3, 4, 5, 6, 1)
            self.axis = (4, 2, 3, 1, 0, 5)

    class TestXPUTransposeCase5(TestTransposeOp):
        def set_case(self):
            self.shape = (2, 16, 96)
            self.axis = (0, 2, 1)

    class TestXPUTransposeCase6(TestTransposeOp):
        def set_case(self):
            self.shape = (2, 10, 12, 16)
            self.axis = (3, 1, 2, 0)

    class TestXPUTransposeCase7(TestTransposeOp):
        def set_case(self):
            self.shape = (2, 10, 2, 16)
            self.axis = (0, 1, 3, 2)

    class TestXPUTransposeCase8(TestTransposeOp):
        def set_case(self):
            self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
            self.axis = (0, 1, 3, 2, 4, 5, 6, 7)

    class TestXPUTransposeCase9(TestTransposeOp):
        def set_case(self):
            self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
            self.axis = (6, 1, 3, 5, 0, 2, 4, 7)


support_types = get_xpu_op_support_types("transpose")
for stype in support_types:
    create_test_class(globals(), XPUTestTransposeOp, stype)

if __name__ == "__main__":
    unittest.main()
