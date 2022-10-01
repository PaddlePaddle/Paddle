#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")

import paddle

from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestSignOP(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'sign'
        self.use_dynamic_create_class = False

    class TestSignOPBase(XPUOpTest):

        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            self.op_type = 'sign'
            self.dtype = self.in_type
            self.init_config()
            self.x = np.random.uniform(-10, 10,
                                       self.input_shape).astype(self.dtype)
            self.inputs = {'X': self.x}
            self.outputs = {'Out': np.sign(self.x)}
            self.attrs = {'use_xpu': True}

        def init_dtype(self):
            self.dtype = np.float32

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_config(self):
            self.input_shape = [864]

    class XPUTestSign1(TestSignOPBase):

        def init_config(self):
            self.input_shape = [2, 768]

    class XPUTestSign2(TestSignOPBase):

        def init_config(self):
            self.input_shape = [3, 8, 4096]

    class XPUTestSign3(TestSignOPBase):

        def init_config(self):
            self.input_shape = [1024]

    class XPUTestSign4(TestSignOPBase):

        def init_config(self):
            self.input_shape = [2, 2, 255]


support_types = get_xpu_op_support_types('sign')
for stype in support_types:
    create_test_class(globals(), XPUTestSignOP, stype)

if __name__ == "__main__":
    unittest.main()
