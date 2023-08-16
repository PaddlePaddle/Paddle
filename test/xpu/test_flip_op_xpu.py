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
from eager_op_test import OpTest
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestFlipOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'flip'
        self.use_dynamic_create_class = False

    class TestFlipOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "flip"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_test_case()
            self.init_attrs()
            x = np.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
            out = np.flip(x, self.axis)
            self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {"Out": out}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def init_attrs(self):
            self.attrs = {"axis": self.axis}

        def init_test_case(self):
            self.in_shape = (3, 2, 2)
            self.axis = [1]

    class TestFlipOp1(TestFlipOp):
        def init_test_case(self):
            self.in_shape = (3, 2, 2, 10)
            self.axis = [3]

    class TestFlipOp2(TestFlipOp):
        def init_test_case(self):
            self.in_shape = (4, 5, 6)
            self.axis = [0, 1]

    class  TestFlipOp3(TestFlipOp):
        def init_test_case(self):
            self.in_shape = (3, 2, 3)
            self.axis = [1, 2]
    
    class TestFlipOp4(TestFlipOp):
        def init_test_case(self):
            self.in_shape = (3, 2, 2, 3)
            self.axis = [2, 3]


support_types = get_xpu_op_support_types('flip')
for stype in support_types:
    create_test_class(globals(), XPUTestFlipOp, stype)

if __name__ == "__main__":
    unittest.main()
