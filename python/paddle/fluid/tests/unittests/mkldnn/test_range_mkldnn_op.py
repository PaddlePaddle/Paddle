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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool
import paddle
import paddle.fluid as fluid


@OpTestTool.skip_if_not_cpu()
class TestRangeOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "range"
        self.set_inputs()

        self.inputs = {
            'Start': np.array([self.start]).astype("float32"),
            'End': np.array([self.end]).astype("float32"),
            'Step': np.array([self.step]).astype("float32")
        }

        self.attrs = {'use_mkldnn': True}

        self.outputs = {
            'Out': np.arange(self.start, self.end, self.step).astype("float32")
        }

    def set_inputs(self):
        self.start = 1.5
        self.end = 300.0
        self.step = 1.5

    def test_check_output(self):
        self.check_output()


class TestRangeNegativeStepOneDNNOp(TestRangeOneDNNOp):
    def set_inputs(self):
        self.start = 1.5
        self.end = -200.0
        self.step = -1.5


class TestRangeNoPrimitiveExecutionOneDNNOp(TestRangeOneDNNOp):
    def set_inputs(self):
        self.start = 0.0
        self.end = 16.0
        self.step = 1.0


class TestRangeOneElementPrimitiveExecutionOneDNNOp(TestRangeOneDNNOp):
    def set_inputs(self):
        self.start = 0.0
        self.end = 17.0
        self.step = 1.0


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
