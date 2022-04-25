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
sys.path.append("..")
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


@skip_check_grad_ci(reason="XPU does not support grad op currently")
class XPUTestElementwisePowOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elementwise_pow'
        self.use_dynamic_create_class = False

    class TestElementwisePowOp(XPUOpTest):
        def setUp(self):
            self.op_type = "elementwise_pow"
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True
            self.compute_input_output()

        def compute_input_output(self):
            self.inputs = {
                'X': np.random.uniform(1, 2, [20, 5]).astype(self.dtype),
                'Y': np.random.uniform(1, 2, [20, 5]).astype(self.dtype)
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestElementwisePowOp_big_shape_1(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X': np.random.uniform(1, 2, [10, 10]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [10, 10]).astype(self.dtype)
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOp_big_shape_2(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X': np.random.uniform(1, 2, [10, 10]).astype(self.dtype),
                'Y': np.random.uniform(0.2, 2, [10, 10]).astype(self.dtype)
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast.")
    class TestElementwisePowOp_scalar(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [3, 3, 4]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [1]).astype(self.dtype)
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOp_tensor(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [100]).astype(self.dtype),
                'Y': np.random.uniform(1, 3, [100]).astype(self.dtype)
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOp_broadcast_0(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [2, 1, 100]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [100]).astype(self.dtype)
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOp_broadcast_1(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [2, 100, 1]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [100]).astype(self.dtype)
            }
            self.attrs = {'axis': 1}
            self.outputs = {
                'Out':
                np.power(self.inputs['X'], self.inputs['Y'].reshape(100, 1))
            }

    class TestElementwisePowOp_broadcast_2(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [100, 3, 1]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [100]).astype(self.dtype)
            }
            self.attrs = {'axis': 0}
            self.outputs = {
                'Out':
                np.power(self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1))
            }

    class TestElementwisePowOp_broadcast_3(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X':
                np.random.uniform(0.1, 1, [2, 20, 5, 1]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [20, 5]).astype(self.dtype)
            }
            self.attrs = {'axis': 1}
            self.outputs = {
                'Out': np.power(self.inputs['X'],
                                self.inputs['Y'].reshape(1, 20, 5, 1))
            }

    class TestElementwisePowOp_broadcast_4(TestElementwisePowOp):
        def compute_input_output(self):
            self.inputs = {
                'X':
                np.random.uniform(0.1, 1, [2, 10, 3, 5]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [2, 10, 1, 5]).astype(self.dtype)
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    class TestElementwisePowOpInt(OpTest):
        def setUp(self):
            self.op_type = "elementwise_pow"
            self.inputs = {
                'X': np.asarray([1, 3, 6]),
                'Y': np.asarray([1, 1, 1])
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

        def test_check_output(self):
            self.check_output()


support_types = get_xpu_op_support_types('elementwise_pow')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwisePowOp, stype)

if __name__ == '__main__':
    unittest.main()
