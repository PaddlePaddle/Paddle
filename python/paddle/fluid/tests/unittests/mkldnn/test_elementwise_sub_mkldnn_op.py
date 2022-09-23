#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
from paddle import enable_static
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
from paddle.fluid.framework import _current_expected_place
import paddle.fluid.core as core


@OpTestTool.skip_if(not (isinstance(_current_expected_place(), core.CPUPlace)),
                    "GPU is not supported")
class TestMKLDNNElementwiseSubOp(OpTest):

    def setUp(self):
        self.op_type = "elementwise_sub"
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
        self.out = np.subtract(self.x, self.y)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ignore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ignore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'))

    def init_axis(self):
        self.axis = -1

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestMKLDNNElementwiseSubOp2(TestMKLDNNElementwiseSubOp):

    def init_input_output(self):
        self.x = np.random.random((100, )).astype(self.dtype)
        self.y = np.random.random((100, )).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)


class TestMKLDNNElementwiseSubOp3(TestMKLDNNElementwiseSubOp):

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)


class TestMKLDNNElementwiseSubOp4(TestMKLDNNElementwiseSubOp):

    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 32]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [4, 32]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)


class TestMKLDNNElementwiseSubOp40(TestMKLDNNElementwiseSubOp):

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 2, [180, 1]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [1, 256]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ignore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ignore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'))


class TestMKLDNNElementwiseSubOp5(TestMKLDNNElementwiseSubOp):

    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)


class TestMKLDNNElementwiseSubOp_broadcast(TestMKLDNNElementwiseSubOp):

    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12, 3).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x - self.y.reshape(1, 10, 12, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseSubOp_xsize_lessthan_ysize_sub(TestMKLDNNElementwiseSubOp):

    def init_input_output(self):
        self.x = np.random.rand(10, 12).astype(self.dtype)
        self.y = np.random.rand(2, 2, 10, 12).astype(self.dtype)
        self.out = self.x - self.y

    def init_axis(self):
        self.axis = 2

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_y(self):
        pass

    def test_check_grad_ignore_x(self):
        pass


@OpTestTool.skip_if_not_cpu_bf16()
class TestBf16(TestMKLDNNElementwiseSubOp):

    def setUp(self):
        self.op_type = "elementwise_sub"
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
        self.x = np.random.random(100, ).astype(self.dtype)
        self.y = np.random.random(100, ).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad_normal(self):
        self.check_grad_with_place(core.CPUPlace(), ["X", "Y"],
                                   "Out",
                                   user_defined_grads=[self.x, -self.x],
                                   user_defined_grad_outputs=[self.x_bf16])

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(core.CPUPlace(), ["Y"],
                                   "Out",
                                   user_defined_grads=[-self.y],
                                   user_defined_grad_outputs=[self.y_bf16])

    def test_check_grad_ignore_y(self):
        self.check_grad_with_place(core.CPUPlace(), ["X"],
                                   "Out",
                                   user_defined_grads=[self.x],
                                   user_defined_grad_outputs=[self.x_bf16])


class TestBf16Broadcasting(TestBf16):

    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def compute_reduced_gradients(self, out_grads):
        part_sum = np.add.reduceat(out_grads, [0], axis=0)
        part_sum = np.add.reduceat(part_sum, [0], axis=1)
        part_sum = np.add.reduceat(part_sum, [0], axis=2)
        return -part_sum.flatten()

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            core.CPUPlace(), ["X", "Y"],
            "Out",
            user_defined_grads=[self.x,
                                self.compute_reduced_gradients(self.x)],
            user_defined_grad_outputs=[self.x_bf16])

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(
            core.CPUPlace(), ["Y"],
            "Out",
            user_defined_grads=[self.compute_reduced_gradients(self.x)],
            user_defined_grad_outputs=[self.x_bf16])


class TestInt8(TestMKLDNNElementwiseSubOp):

    def init_kernel_type(self):
        self.use_mkldnn = True
        self._cpu_only = True

    def init_dtype(self):
        self.dtype = np.int8

    def init_input_output(self):
        self.x = np.random.randint(0, 3, (12, 9)).astype("int8")
        self.y = np.random.randint(0, 3, (12, 9)).astype("int8")
        self.out = np.subtract(self.x, self.y)

    def init_scales(self):
        self.attrs['Scale_x'] = 1.0
        self.attrs['Scale_y'] = 1.0
        self.attrs['Scale_out'] = 1.0

    def test_check_output(self):
        self.init_scales()
        self.check_output()

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass

    def test_check_grad_ignore_y(self):
        pass


if __name__ == '__main__':
    enable_static()
    unittest.main()
