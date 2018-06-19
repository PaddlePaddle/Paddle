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
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
from test_elementwise_add_op import TestElementwiseAddOp


class TestMKLDNNElementwiseAddOp(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_scalar(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


class TestMKLDNNElementwiseAddOp_scalar2(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


class TestMKLDNNElementwiseAddOp_Vector(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((32, )).astype(self.dtype)
        self.y = np.random.random((32, )).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TesMKLDNNtElementwiseAddOp_broadcast_0(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2).astype(self.dtype)
        self.out = self.x + self.y.reshape(2, 1, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestMKLDNNElementwiseAddOp_broadcast_1(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(3).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 1, 1)

    def init_axis(self):
        self.axis = 1


class TestMKLDNNElementwiseAddOp_broadcast_2(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(5).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 1, 5)


class TestMKLDNNElementwiseAddOp_broadcast_3(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 4, 1)

    def init_axis(self):
        self.axis = 1


class TestMKLDNNElementwiseAddOp_broadcast_4(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(2, 1, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestMKLDNNElementwiseAddOp_rowwise_add_0(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 3, 4)

    def init_axis(self):
        self.axis = 2


class TestMKLDNNElementwiseAddOp_rowwise_add_1(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1)

    def init_axis(self):
        self.axis = 1


class TestMKLDNNElementwiseAddOp_channelwise_add(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 20, 20).astype(self.dtype)
        self.y = np.random.rand(2, 1, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


if __name__ == '__main__':
    unittest.main()
