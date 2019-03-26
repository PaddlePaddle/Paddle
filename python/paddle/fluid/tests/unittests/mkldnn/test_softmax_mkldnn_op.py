#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.test_softmax_op import TestSoftmaxOp, stable_softmax
from mkldnn_op_test import check_if_mkldnn_primitives_exist_in_bwd


class TestSoftmaxMKLDNNOp(TestSoftmaxOp):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestSoftmaxMKLDNNOp2(TestSoftmaxMKLDNNOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]


# Check if primitives already exist in backward
class TestSoftmaxMKLDNNPrimitivesAlreadyExist(unittest.TestCase):
    def setUp(self):
        super(TestSoftmaxMKLDNNPrimitivesAlreadyExist, self).setUp()

        np.random.seed(123)
        self.op_type = 'softmax'
        self.x = np.random.uniform(-1, 1, 2).astype(np.float32)
        self.out = stable_softmax(self.x)
        self.out_grad = np.random.random_sample(self.x.shape).astype(np.float32)
        self.x_grad = self.__softmax_bwd(self.out, self.out_grad)

    # Softmax grad calculation
    def __softmax_bwd(self, out, out_grad):
        return out * (out_grad - np.dot(out, out_grad))

    def test_check(self):
        check_if_mkldnn_primitives_exist_in_bwd(
            self, self.op_type, self.x, self.out, self.out_grad, self.x_grad)


if __name__ == '__main__':
    unittest.main()
