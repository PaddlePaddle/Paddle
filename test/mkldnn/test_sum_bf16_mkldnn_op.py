# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import convert_float_to_uint16
from test_sum_op import TestSumOp

from paddle import enable_static
from paddle.base import core


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestSumBF16MKLDNN(TestSumOp):
    def setUp(self):
        self.op_type = "sum"
        self.use_mkldnn = True
        self.mkldnn_data_type = "bfloat16"

        # float32 input to be use for reference
        x0 = np.random.random((25, 8)).astype('float32')
        x1 = np.random.random((25, 8)).astype('float32')
        x2 = np.random.random((25, 8)).astype('float32')

        # actual input (bf16) to bf16 sum op
        x0_bf16 = convert_float_to_uint16(x0)
        x1_bf16 = convert_float_to_uint16(x1)
        x2_bf16 = convert_float_to_uint16(x2)

        self.inputs = {"X": [("x0", x0_bf16), ("x1", x1_bf16), ("x2", x2_bf16)]}

        y = x0 + x1 + x2
        self.outputs = {'Out': convert_float_to_uint16(y)}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)

    def test_check_grad(self):
        pass


if __name__ == '__main__':
    enable_static()
    unittest.main()
