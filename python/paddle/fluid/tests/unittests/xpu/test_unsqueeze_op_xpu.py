# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")

import numpy as np

import paddle
import paddle.fluid as fluid
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


# Correct: General.
class XPUTestUnsqueezeOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = "unsqueeze"
        self.use_dynamic_create_class = False

    class TestUnsqueezeOp(XPUOpTest):

        def setUp(self):
            self.op_type = "unsqueeze"
            self.__class__.op_type = "unsqueeze"
            self.use_mkldnn = False
            self.init_test_case()
            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype)
            }
            self.init_attrs()
            self.outputs = {"Out": self.inputs["X"].reshape(self.new_shape)}

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            place = paddle.XPUPlace(0)
            if self.dtype == np.bool_:
                return
            else:
                self.check_grad_with_place(place, ['X'], 'Out')

        def init_test_case(self):
            self.ori_shape = (3, 40)
            self.axes = (1, 2)
            self.new_shape = (3, 1, 1, 40)

        def init_attrs(self):
            self.attrs = {"axes": self.axes}

    # Correct: Single input index.
    class TestUnsqueezeOp1(TestUnsqueezeOp):

        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (-1, )
            self.new_shape = (20, 5, 1)

    # Correct: Mixed input axis.
    class TestUnsqueezeOp2(TestUnsqueezeOp):

        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (0, -1)
            self.new_shape = (1, 20, 5, 1)

    # Correct: There is duplicated axis.
    class TestUnsqueezeOp3(TestUnsqueezeOp):

        def init_test_case(self):
            self.ori_shape = (10, 2, 5)
            self.axes = (0, 3, 3)
            self.new_shape = (1, 10, 2, 1, 1, 5)

    # Correct: Reversed axes.
    class TestUnsqueezeOp4(TestUnsqueezeOp):

        def init_test_case(self):
            self.ori_shape = (10, 2, 5)
            self.axes = (3, 1, 1)
            self.new_shape = (10, 1, 1, 2, 5, 1)


support_types = get_xpu_op_support_types("unsqueeze")
for stype in support_types:
    create_test_class(globals(), XPUTestUnsqueezeOp, stype)

if __name__ == "__main__":
    unittest.main()
