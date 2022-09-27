# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import enable_static
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
from paddle.fluid.framework import _current_expected_place
import paddle.fluid.core as core


@OpTestTool.skip_if(not (isinstance(_current_expected_place(), core.CPUPlace)),
                    "GPU is not supported")
class TestMKLDNNElementwiseDivOp(OpTest):

    def setUp(self):
        self.op_type = "elementwise_div"
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', None, 0.005, False, 0.02)

    def test_check_grad_ignore_x(self):
        self.check_grad(['Y'], 'Out', set("X"), 0.005, False, 0.02)

    def test_check_grad_ignore_y(self):
        self.check_grad(['X'], 'Out', set('Y'), 0.005, False, 0.02)

    def init_axis(self):
        self.axis = -1

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestMKLDNNElementwiseDivOp2(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)


class TestMKLDNNElementwiseDivOp3(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)


class TestMKLDNNElementwiseDivOp4(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 32]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [4, 32]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    # TODO(piotrekobiIntel): Enable when grad is ready
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass


class TestMKLDNNElementwiseDivOp5(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    # TODO(piotrekobiIntel): Enable when grad is ready
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass


@OpTestTool.skip_if_not_cpu_bf16()
class TestBf16(TestMKLDNNElementwiseDivOp):

    def setUp(self):
        self.op_type = "elementwise_div"
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.x_bf16 = convert_float_to_uint16(self.x)
        self.y_bf16 = convert_float_to_uint16(self.y)
        self.inputs = {'X': self.x_bf16, 'Y': self.y_bf16}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': convert_float_to_uint16(self.out)}

    def init_dtype(self):
        self.dtype = np.float32
        self.mkldnn_data_type = "bfloat16"

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad_normal(self):
        self.check_grad_with_place(core.CPUPlace(), ["X", "Y"],
                                   "Out",
                                   user_defined_grads=[
                                       np.divide(self.x, self.y),
                                       np.divide((np.multiply(-self.x, self.x)),
                                                 np.multiply(self.y, self.y))
                                   ],
                                   user_defined_grad_outputs=[self.x_bf16])

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(core.CPUPlace(), ["Y"],
                                   "Out",
                                   user_defined_grads=[
                                       np.divide((np.multiply(-self.x, self.y)),
                                                 np.multiply(self.y, self.y))
                                   ],
                                   user_defined_grad_outputs=[self.y_bf16])

    def test_check_grad_ignore_y(self):
        self.check_grad_with_place(
            core.CPUPlace(), ["X"],
            "Out",
            user_defined_grads=[np.divide(self.x, self.y)],
            user_defined_grad_outputs=[self.x_bf16])


class TestBf16Broadcasting(TestBf16):

    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass


if __name__ == '__main__':
    enable_static()
    unittest.main()
