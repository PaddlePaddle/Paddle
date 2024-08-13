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

import unittest

import numpy as np
from op_test import OpTest, OpTestTool

import paddle
from paddle.base import core


@OpTestTool.skip_if_not_cpu_bf16()
class TestShuffleChannelOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "shuffle_channel"
        self.set_dtype()
        self.set_group()
        self.inputs = {'X': np.random.random((5, 64, 2, 3)).astype(self.dtype)}
        self.attrs = {'use_mkldnn': True, 'group': self.group}

        _, c, h, w = self.inputs['X'].shape
        input_reshaped = np.reshape(
            self.inputs['X'], (-1, self.group, c // self.group, h, w)
        )
        input_transposed = np.transpose(input_reshaped, (0, 2, 1, 3, 4))
        self.outputs = {'Out': np.reshape(input_transposed, (-1, c, h, w))}

    def set_dtype(self):
        self.dtype = np.float32

    def set_group(self):
        self.group = 4

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)


class TestShuffleChannelSingleGroupOneDNNOp(TestShuffleChannelOneDNNOp):
    def set_group(self):
        self.group = 1


class TestShuffleChannelBF16OneDNNOp(TestShuffleChannelOneDNNOp):
    def set_dtype(self):
        self.dtype = np.uint16


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
