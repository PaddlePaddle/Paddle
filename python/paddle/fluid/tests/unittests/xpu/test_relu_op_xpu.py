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

import numpy as np
import sys
import unittest
sys.path.append("..")

import paddle

from op_test_xpu import XPUOpTest
from get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
paddle.enable_static()
SEED = 2022


class XPUTestReluOp(XPUOpTestWrapper):
    def __init__(self) -> None:
        self.op_name = "relu"
        self.use_dynamic_create_class = False

    class TestReluOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "relu"
            self.init_dtype()
            self.init_shape()
            self.init_config()
            np.random.seed(SEED)

        def init_shape(self):
            self.shape = (3, 2)

        def init_dtype(self):
            self.dtype = self.in_type

        def init_config(self):
            x = np.random.standard_normal(self.shape).astype(self.dtype)
            self.inputs = {'X': self.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': np.maximum(0, x)}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ["X"], "Out")

    class TestReluOp1(TestReluOp):
        def init_shape(self):
            self.shape = (2)

    class TestReluOp1(TestReluOp):
        def init_shape(self):
            self.shape = (2, 3, 4)

    class TestReluOp1(TestReluOp):
        def init_shape(self):
            self.shape = (2, 3, 4, 4)

    class TestReluOp1(TestReluOp):
        def init_shape(self):
            self.shape = (2, 3, 4, 4, 5)


support_types = get_xpu_op_support_types("relu")
for stype in support_types:
    create_test_class(globals(), XPUTestReluOp, stype)

if __name__ == "__main__":
    unittest.main()
