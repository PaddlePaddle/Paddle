#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.test_elementwise_mul_op import *


# TODO(LeoZhao-Intel): re-enable this case
# https://github.com/PaddlePaddle/Paddle/issues/16764
@unittest.skip("Not supported well on avx2.")
class TestElementwiseMulMKLDNNOp_BroadcastNCHW16c(ElementwiseMulOp):
    def init_input_output(self):
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)
        self.y = np.random.rand(1, 16).astype(self.dtype)

        self.out = x * self.y.reshape(1, 16, 1, 1)

        self.out = self.out.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_BroadcastNCHW16c, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nc"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


@unittest.skip(
    "Not implemented yet.")  # TODO(mgallus): enable when implemented.
class TestElementwiseMulMKLDNNOp_BroadcastNCHW8c(ElementwiseMulOp):
    def init_input_output(self):
        x = np.random.rand(1, 8, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 8, 2, 2)
        self.y = np.random.rand(1, 8).astype(self.dtype)

        self.out = x * self.y.reshape(1, 8, 1, 1)
        self.out = self.out.transpose(0, 2, 3, 1).reshape(1, 8, 2, 2)

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_BroadcastNCHW8c, self).setUp()
        self.attrs["x_data_format"] = "nchw8c"
        self.attrs["y_data_format"] = "nc"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNCHW(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = np.random.rand(1, 16).astype(self.dtype)

        self.out = self.x * self.y.reshape(1, 16, 1, 1)

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNCHW16C(ElementwiseMulOp):
    def init_input_output(self):
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = self.x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNCHW16C, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNoReorders(ElementwiseMulOp):
    def init_input_output(self):
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = self.x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNoReorders, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackWithReorder1(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = self.x * y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackWithReorder1, self).setUp()
        self.attrs["x_data_format"] = "nchw"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackWithReorder2(ElementwiseMulOp):
    def init_input_output(self):
        self.y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackWithReorder2, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNoReorders2(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16).astype(self.dtype)
        self.y = np.random.rand(1, 16).astype(self.dtype)

        self.out = self.x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNoReorders2, self).setUp()
        self.attrs["x_data_format"] = "nc"
        self.attrs["y_data_format"] = "nc"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


if __name__ == '__main__':
    unittest.main()
