#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import _current_expected_place


def ref_softplus(x, beta, threshold):
    x_beta = beta * x
    out = np.select([x_beta <= threshold, x_beta > threshold],
                    [np.log(1 + np.exp(x_beta)) / beta, x])
    return out


@OpTestTool.skip_if_not_cpu_bf16()
class TestSoftplusOneDNNOp(OpTest):

    def setUp(self):
        self.op_type = "softplus"
        self.beta = 1
        self.threshold = 20
        self.config()
        self.set_dtype()
        self.attrs = {'use_mkldnn': True, 'beta': self.beta}
        self.x = np.random.random(self.x_shape)
        self.out = ref_softplus(self.x, self.beta, self.threshold)

        if self.dtype != np.float32:
            self.x = convert_float_to_uint16(self.x)

        self.inputs = {'X': self.out}
        self.outputs = {
            'Out': ref_softplus(self.out, self.beta, self.threshold)
        }

    def config(self):
        self.x_shape = (10, 10)

    def set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestSoftplus4DOneDNNOp(TestSoftplusOneDNNOp):

    def config(self):
        self.x_shape = (10, 5, 4, 2)


class TestSoftplus6DOneDNNOp(TestSoftplusOneDNNOp):

    def config(self):
        self.x_shape = (3, 2, 2, 5, 4, 2)


class TestSoftplus6DExtendedFunctorOneDNNOp(TestSoftplusOneDNNOp):

    def config(self):
        self.x_shape = (3, 5, 2, 5, 4, 2)
        self.beta = 2.5


class TestSoftplus3DExtendedFunctorOneDNNOp(TestSoftplusOneDNNOp):

    def config(self):
        self.x_shape = (20, 4, 2)
        self.beta = 0.4


class TestSoftplusBF16OneDNNOp(TestSoftplusOneDNNOp):

    def set_dtype(self):
        self.dtype = np.uint16


class TestSoftplus4DBF16OneDNNOp(TestSoftplus4DOneDNNOp):

    def set_dtype(self):
        self.dtype = np.uint16


class TestSoftplus6DBF16OneDNNOp(TestSoftplus6DOneDNNOp):

    def set_dtype(self):
        self.dtype = np.uint16


class TestSoftplus3DExtendedFunctorBF16OneDNNOp(
        TestSoftplus3DExtendedFunctorOneDNNOp):

    def set_dtype(self):
        self.dtype = np.uint16


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
