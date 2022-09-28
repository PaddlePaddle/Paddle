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

import sys

sys.path.append("..")
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestStackOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'stack'
        self.use_dynamic_create_class = False

    @skip_check_grad_ci(reason="There is no grad kernel for stack_xpu op.")
    class TestStackOp(XPUOpTest):

        def initDefaultParameters(self):
            self.num_inputs = 4
            self.input_dim = (5, 6, 7)
            self.axis = 0
            self.dtype = np.float32

        def setUp(self):
            self.initDefaultParameters()
            self.initParameters()
            self.__class__.use_xpu = True
            self.__class__.op_type = 'stack'
            self.x = []
            for i in range(self.num_inputs):
                self.x.append(
                    np.random.random(size=self.input_dim).astype(self.dtype))

            tmp = []
            x_names = self.get_x_names()
            for i in range(self.num_inputs):
                tmp.append((x_names[i], self.x[i]))

            self.inputs = {'X': tmp}
            self.outputs = {'Y': np.stack(self.x, axis=self.axis)}
            self.attrs = {'axis': self.axis}

        def init_dtype(self):
            self.dtype = self.in_type

        def initParameters(self):
            pass

        def get_x_names(self):
            x_names = []
            for i in range(self.num_inputs):
                x_names.append('x{}'.format(i))
            return x_names

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            if self.dtype == np.int32 or self.dtype == np.int64:
                pass
            else:
                self.check_grad_with_place(paddle.XPUPlace(0),
                                           self.get_x_names(), 'Y')

    class TestStackOp1(TestStackOp):

        def initParameters(self):
            self.num_inputs = 16

    class TestStackOp2(TestStackOp):

        def initParameters(self):
            self.num_inputs = 30

    class TestStackOp3(TestStackOp):

        def initParameters(self):
            self.axis = -1

        def test_check_grad(self):
            pass

    class TestStackOp4(TestStackOp):

        def initParameters(self):
            self.axis = -4

        def test_check_grad(self):
            pass

    class TestStackOp5(TestStackOp):

        def initParameters(self):
            self.axis = 1

    class TestStackOp6(TestStackOp):

        def initParameters(self):
            self.axis = 3

    class TestStackOp7(TestStackOp):

        def initParameters(self):
            self.num_inputs = 4
            self.input_dim = (5, 6, 7)
            self.axis = 0
            self.dtype = np.int64

        def test_check_grad(self):
            pass

    class TestStackOp8(TestStackOp):

        def initParameters(self):
            self.num_inputs = 4
            self.input_dim = (5, 6, 7)
            self.axis = 0
            self.dtype = np.int32

        def test_check_grad(self):
            pass


support_types = get_xpu_op_support_types('stack')
for stype in support_types:
    create_test_class(globals(), XPUTestStackOp, stype)

if __name__ == "__main__":
    unittest.main()
