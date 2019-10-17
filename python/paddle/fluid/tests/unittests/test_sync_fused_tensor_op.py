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
from op_test import OpTest
from paddle.fluid import core

alignment = 256


class TestSyncFusedTensorOp(OpTest):
    def setUp(self):
        self.op_type = "sync_fused_tensor"
        self.dtype = np.float32
        self.Input = self.init_input()
        self.FusedInput = self.init_fused_inout(self.Input)
        self.FusedOutput = self.init_fused_inout(self.Input)
        self.inputs = {"Input": self.Input, "FusedInput": self.FusedInput}
        self.outputs = {"FusedOutput": self.FusedOutput}

    def init_input(self):
        inputs = []
        inputs.append(("x1", np.random.random([20, 3]).astype(self.dtype)))
        inputs.append(("x2", np.random.random([20]).astype(self.dtype)))
        inputs.append(("x3", np.random.random([1]).astype(self.dtype)))
        inputs.append(("x4", np.random.random([200, 30]).astype(self.dtype)))
        inputs.append(("x5", np.random.random([30]).astype(self.dtype)))
        inputs.append(("x6", np.random.random([1]).astype(self.dtype)))
        return inputs

    def init_fused_inout(self, input_list):
        inputs = []
        for input in input_list:
            length = len(input[1].flatten())
            aligned_len = (length + alignment) / alignment * alignment
            out = np.zeros(int(aligned_len)).astype(self.dtype)
            out[0:length] = input[1].flatten()
            inputs.append(out)
        coalesce_tensor_var = np.concatenate([input for input in inputs])
        return coalesce_tensor_var

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(
                place=core.CUDAPlace(0),
                no_check_set=["FusedOutput"],
                atol=1e-5)


if __name__ == '__main__':
    unittest.main()
