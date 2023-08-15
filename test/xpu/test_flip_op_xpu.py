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


class XPUTestFlipOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'flip'
        self.use_dynamic_create_class = False

    class TestFlipOp(XPUOpTest):
        def setUp(self):
            self.op_type = "flip"
            self.set_xpu()
            self.place = paddle.XPUPlace(0)
            self.init_test_case()
            self.dtype = self.in_type
            self.inputs = {
                "X": np.random.random(self.in_shape).astype(self.dtype)
            }
            self.init_attrs()
            self.outputs = {"Out": self.inputs["X"]}
  
        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if self.no_need_check_grad:
                return
            else:
                self.check_grad_with_place(
                    self.place, ['X'], 'Out'
                )

        def init_test_case(self):
            self.in_shape = (3, 2, 2, 10)
            self.axis = 1

        def init_attrs(self):
            self.attrs = {"axis": self.axis}

support_types = get_xpu_op_support_types('flip')
for stype in support_types:
    create_test_class(globals(), XPUTestFlipOp, stype)

if __name__ == "__main__":
    unittest.main()
