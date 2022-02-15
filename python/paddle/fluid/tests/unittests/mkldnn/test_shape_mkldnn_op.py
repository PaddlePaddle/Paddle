#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core
from paddle.fluid.op import Operator


@OpTestTool.skip_if_not_cpu_bf16()
class TestShape3DFP32OneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "shape"
        self.config()
        self.attrs = {'use_mkldnn': True}
        self.inputs = {'Input': np.zeros(self.shape).astype(self.dtype)}
        self.outputs = {'Out': np.array(self.shape)}

    def config(self):
        self.shape = [5, 7, 4]
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())


class TestShape6DBF16OneDNNOp(TestShape3DFP32OneDNNOp):
    def config(self):
        self.shape = [10, 2, 3, 4, 5, 2]
        self.dtype = np.uint16


class TestShape9DINT8OneDNNOp(TestShape3DFP32OneDNNOp):
    def config(self):
        self.shape = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.dtype = np.int8


class TestShape2DUINT8OneDNNOp(TestShape3DFP32OneDNNOp):
    def config(self):
        self.shape = [7, 11]
        self.dtype = np.uint8


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
