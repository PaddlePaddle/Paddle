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

import numpy as np
import unittest
import sys

sys.path.append("..")
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.fluid as fluid
from test_activation_op import ref_swish, expit

paddle.enable_static()
SEED = 1024


class TestSwishOp(OpTest):

    def setUp(self):
        self.op_type = "swish"
        self.set_npu()
        self.init_dtype()
        np.random.seed(2048)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = ref_swish(x)
        self.inputs = {'X': x}
        self.attrs = {'beta': 1.0}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        beta = self.attrs['beta']
        out = self.outputs['Out']
        x = self.inputs['X']
        dx = beta * out + expit(x) * (1 - beta * out)
        dx = dx / x.size

        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   max_relative_error=0.01,
                                   user_defined_grads=[dx])

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32


class TestSwishOpFp16(TestSwishOp):

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
