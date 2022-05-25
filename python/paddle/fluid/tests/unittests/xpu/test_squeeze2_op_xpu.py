#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")

import numpy as np

from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
import paddle

paddle.enable_static()


class XPUTestSqueeze2Op(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "squeeze2"
        self.use_dynamic_create_class = False

    class TestSqueeze2Op(XPUOpTest):
        def setUp(self):
            self.op_type = "squeeze2"
            self.use_mkldnn = False
            self.init_dtype()
            self.init_test_case()
            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype)
            }
            self.outputs = {
                "Out": self.inputs["X"].reshape(self.new_shape),
                "XShape": np.random.random(self.ori_shape).astype(self.dtype)
            }
            self.init_attrs()

        def init_dtype(self):
            self.dtype = self.in_type

        def init_attrs(self):
            self.attrs = {"axes": self.axes}

        def init_test_case(self):
            self.ori_shape = (1, 3, 1, 40)
            self.axes = (0, 2)
            self.new_shape = (3, 40)

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            place = paddle.XPUPlace(0)
            if self.dtype in [np.float32, np.float64]:
                self.check_grad_with_place(place, ['X'], 'Out')
            elif self.dtype == np.bool_:
                return
            else:
                user_defined_grad_outputs = np.random.random(
                    self.new_shape).astype(self.dtype)
                self.check_grad_with_place(
                    place, ['X'],
                    'Out',
                    user_defined_grad_outputs=user_defined_grad_outputs)

    # Correct: There is mins axis.
    class TestSqueeze2Op1(TestSqueeze2Op):
        def init_test_case(self):
            self.ori_shape = (1, 20, 1, 5)
            self.axes = (0, -2)
            self.new_shape = (20, 5)

    # Correct: No axes input.
    class TestSqueeze2Op2(TestSqueeze2Op):
        def init_test_case(self):
            self.ori_shape = (1, 20, 1, 5)
            self.axes = ()
            self.new_shape = (20, 5)

    # Correct: Just part of axes be squeezed. 
    class TestSqueeze2Op3(TestSqueeze2Op):
        def init_test_case(self):
            self.ori_shape = (6, 1, 5, 1, 4, 1)
            self.axes = (1, -1)
            self.new_shape = (6, 5, 1, 4)


support_types = get_xpu_op_support_types("squeeze2")
for stype in support_types:
    create_test_class(globals(), XPUTestSqueeze2Op, stype)

if __name__ == "__main__":
    unittest.main()
