#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class XPUTestReciprocalOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reciprocal'
        self.use_dynamic_create_class = False

    class TestReciprocalOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "reciprocal"
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_shape()
            self.init_data()
            self.outputs = {'Out': np.reciprocal(self.inputs['X'])}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_shape(self):
            self.shape = (4, 10, 10)

        def init_data(self):
            x = np.random.random(self.shape).astype("float32")
            self.inputs['X'] = x

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if core.is_compiled_with_xpu():
                self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestReciprocalOp1(TestReciprocalOp):
        def init_shape(self):
            self.shape = (8, 16, 8)

    class TestReciprocalOp2(TestReciprocalOp):
        def init_shape(self):
            self.shape = (8, 16)

    class TestReciprocalOp3(TestReciprocalOp):
        def init_shape(self):
            self.shape = (4, 8, 16)

    class TestReciprocalOp4(TestReciprocalOp):
        def init_shape(self):
            self.shape = (4, 8, 8)

    class TestReciprocalOp5(TestReciprocalOp):
        def init_shape(self):
            self.shape = (4, 8, 16)


support_types = get_xpu_op_support_types('reciprocal')
for stype in support_types:
    create_test_class(globals(), XPUTestReciprocalOp, stype)

if __name__ == '__main__':
    unittest.main()
