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
from paddle.fluid.tests.unittests.test_elementwise_add_op import *
'''
MKLDNN does not support tensors of dimensions number equal to 3.
Such dimensions cause exceptions in MKLDNN reorder primitive.
The DNNL-based kernel is used only when broadcasting is not required
(see GetExpectedKernelType() methods in elementwise_add_op.h).
'''


class TestMKLDNNElementwiseAddOp(TestElementwiseAddOp):
    def init_data_format(self):
        self.data_format = 'MKLDNN'

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNElementwiseAddOp2(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((100, )).astype(self.dtype)
        self.y = np.random.random((100, )).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestMKLDNNElementwiseAddOp3(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.add(self.x, self.y)


if __name__ == '__main__':
    unittest.main()
