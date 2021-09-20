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
from paddle.fluid.tests.unittests.test_elementwise_sub_op import TestElementwiseSubOp
from paddle import enable_static


class TestMKLDNNElementwiseSubOp(TestElementwiseSubOp):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_dtype(self):
        self.dtype = np.float32

    # # TODO(piotrekobiIntel): Enable when grad is ready
    # def test_check_grad_normal(self):
    #     pass

    # def test_check_grad_ingore_x(self):
    #     pass

    # def test_check_grad_ingore_y(self):
    #     pass


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


# class TestMKLDNNElementwiseSubOp4(TestMKLDNNElementwiseSubOp):
#     def init_input_output(self):
#         self.x = np.random.uniform(1, 2, [2, 3, 4, 32]).astype(self.dtype)
#         self.y = np.random.uniform(1, 2, [4, 32]).astype(self.dtype)
#         self.out = np.subtract(self.x, self.y)

# class TestMKLDNNElementwiseSubOp5(TestMKLDNNElementwiseSubOp):
#     def init_input_output(self):
#         self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
#         self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
#         self.out = np.subtract(self.x, self.y)


class TestMKLDNNElementwiseSubOp_broadcast_3(TestMKLDNNElementwiseSubOp):
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

    # TODO(piotrekobiIntel): Enable when grad is ready
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_y(self):
        pass

    def test_check_grad_ingore_x(self):
        pass


class TestInt8(TestElementwiseSubOp):
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
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.init_scales()
        self.check_output()

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


if __name__ == '__main__':
    enable_static()
    unittest.main()
