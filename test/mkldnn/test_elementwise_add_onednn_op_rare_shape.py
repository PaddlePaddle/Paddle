#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTestTool
from test_elementwise_add_op import TestElementwiseAddOp

from paddle import enable_static
from paddle.base import core
from paddle.base.framework import _current_expected_place


@OpTestTool.skip_if(
    not (isinstance(_current_expected_place(), core.CPUPlace)),
    "GPU is not supported",
)
# Special cases for swin transformer, will ignore grad check
class TestOneDNNElementwiseAddSrcDifferentShape(TestElementwiseAddOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.check_pir_onednn = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_y(self):
        pass

    def test_check_grad_ignore_x(self):
        pass

    def init_input_output(self):
        self.x = np.random.random((1, 4, 16, 12, 12)).astype(self.dtype)
        self.y = np.random.random((1, 4, 1, 12, 12)).astype(self.dtype)
        self.out = np.add(self.x, self.y)


if __name__ == '__main__':
    enable_static()
    unittest.main()
