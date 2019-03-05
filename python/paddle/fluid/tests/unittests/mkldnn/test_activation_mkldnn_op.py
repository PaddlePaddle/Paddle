#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_activation_op import TestRelu, TestTanh, TestSqrt, TestAbs
from mkldnn_op_test import check_if_mkldnn_primitives_exist_in_bwd


class TestMKLDNNReluDim2(TestRelu):
    def setUp(self):
        super(TestMKLDNNReluDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}


class TestMKLDNNTanhDim2(TestTanh):
    def setUp(self):
        super(TestMKLDNNTanhDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}


class TestMKLDNNSqrtDim2(TestSqrt):
    def setUp(self):
        super(TestMKLDNNSqrtDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}


class TestMKLDNNAbsDim2(TestAbs):
    def setUp(self):
        super(TestMKLDNNAbsDim2, self).setUp()
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNReluDim4(TestRelu):
    def setUp(self):
        super(TestMKLDNNReluDim4, self).setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNTanhDim4(TestTanh):
    def setUp(self):
        super(TestMKLDNNTanhDim4, self).setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.tanh(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNSqrtDim4(TestSqrt):
    def setUp(self):
        super(TestMKLDNNSqrtDim4, self).setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.sqrt(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNAbsDim4(TestAbs):
    def setUp(self):
        super(TestMKLDNNAbsDim4, self).setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        self.inputs = {'X': x}
        self.outputs = {'Out': np.abs(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}


# Check if primitives already exist in backward
class TestMKLDNNAbsPrimitivesAlreadyExist(unittest.TestCase):
    def setUp(self):
        super(TestMKLDNNAbsPrimitivesAlreadyExist, self).setUp()

        np.random.seed(123)
        self.op_type = 'abs'
        self.x = np.random.uniform(-1, 1, [2, 2]).astype(np.float32)
        self.out = np.abs(self.x)
        self.out_grad = np.random.random_sample(self.x.shape).astype(np.float32)
        self.x_grad = self.__abs_bwd(self.x, self.out_grad)

    # Abs grad calculation
    def __abs_bwd(self, x, out_grad):
        return out_grad * np.sign(x)

    def test_check(self):
        check_if_mkldnn_primitives_exist_in_bwd(
            self, self.op_type, self.x, self.out, self.out_grad, self.x_grad)


if __name__ == '__main__':
    unittest.main()
