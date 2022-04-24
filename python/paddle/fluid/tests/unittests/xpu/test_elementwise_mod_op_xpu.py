#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append("")
import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard

import paddle
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestElementwiseModOp(XPUOpTestWrapper):
    def __init__(self) -> None:
        self.op_name = 'elementwise_mod'
        self.use_dynamic_create_class = False

    class ElementwiseModOp(XPUOpTest):
        def init_kernel_type(self):
            self.use_mkldnn = False

        def init_input_output(self):
            self.x = np.random.uniform(1, 100, [13, 17]).astype(self.dtype)
            self.y = np.random.uniform(1, 100, [13, 17]).astype(self.dtype)
            self.out = np.mod(self.x, self.y)
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(self.x),
                'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
            }
            self.outputs = {'Out': self.out}
            self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

        def init_dtype(self):
            pass

        def init_axis(self):
            pass

        def setUp(self):
            self.op_type = 'elementwise_mod'
            self.use_xpu = True
            self.dtype = self.in_type
            self.axis = -1
            self.init_dtype()
            self.init_input_output()
            self.init_kernel_type()
            self.init_axis()

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestElementwiseModOp_broadcast_1(ElementwiseModOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 100, 3).astype(self.dtype),
                'Y': np.random.rand(2, 100, 3).astype(self.dtype)
            }

            self.attrs = {'axis': 1}
            self.outputs = {'Out': self.inputs['X'] % self.inputs['Y']}

    class TestElementwiseModOp_broadcast_1(ElementwiseModOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(22, 128, 3).astype(self.dtype),
                'Y': np.random.rand(22, 128, 3).astype(self.dtype)
            }

            self.attrs = {'axis': 1}
            self.outputs = {'Out': self.inputs['X'] % self.inputs['Y']}


support_types = get_xpu_op_support_types('elementwise_mod')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseModOp, stype)

if __name__ == '__main__':
    unittest.main()
