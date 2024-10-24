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
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestRangeOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "range"
        self.use_dynamic_create_class = False

    class TestRangeOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "range"
            self.init_dtype()
            self.init_config()
            start = np.array([self.case[0]]).astype(
                self.dtype if self.dtype != np.uint16 else np.float32
            )
            end = np.array([self.case[1]]).astype(
                self.dtype if self.dtype != np.uint16 else np.float32
            )
            step = np.array([self.case[2]]).astype(
                self.dtype if self.dtype != np.uint16 else np.float32
            )
            if self.dtype == np.uint16:
                start = convert_float_to_uint16(start)
                end = convert_float_to_uint16(end)
                step = convert_float_to_uint16(step)
            self.inputs = {'Start': start, 'End': end, 'Step': step}

            self.outputs = {
                'Out': np.arange(
                    self.case[0], self.case[1], self.case[2]
                ).astype(self.dtype)
            }

        def set_xpu(self):
            self.__class__.no_need_check_grad = True

        def init_dtype(self):
            self.dtype = self.in_type

        def init_config(self):
            self.case = (0, 1, 0.2) if self.dtype == np.float32 else (0, 5, 1)

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, check_dygraph=False)

    class TestRangeOpCase0(TestRangeOp):
        def init_config(self):
            self.case = (0, 5, 1)

    class TestRangeOpCase1(TestRangeOp):
        def init_config(self):
            self.case = (0, 5, 2)

    class TestRangeOpCase2(TestRangeOp):
        def init_config(self):
            self.case = (10, 1, -2)

    class TestRangeOpCase3(TestRangeOp):
        def init_config(self):
            self.case = (-1, -10, -2)

    class TestRangeOpCase4(TestRangeOp):
        def init_config(self):
            self.case = (10, -10, -11)

    class TestRangeOpCase5(TestRangeOp):
        def init_config(self):
            self.case = (0, 1, 1)


support_types = get_xpu_op_support_types("range")
for stype in support_types:
    create_test_class(globals(), XPUTestRangeOp, stype)

if __name__ == "__main__":
    unittest.main()
