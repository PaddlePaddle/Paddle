# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import math
from op_test import OpTest
import paddle.fluid.core as core


class TestShuffleChannelOp(OpTest):

    def setUp(self):
        self.op_type = "shuffle_channel"
        self.batch_size = 10
        self.input_channels = 16
        self.layer_h = 4
        self.layer_w = 4
        self.group = 4
        self.x = np.random.random(
            (self.batch_size, self.input_channels, self.layer_h,
             self.layer_w)).astype('float32')
        self.inputs = {'X': self.x}
        self.attrs = {'group': self.group}
        n, c, h, w = self.x.shape
        input_reshaped = np.reshape(self.x,
                                    (-1, self.group, c // self.group, h, w))
        input_transposed = np.transpose(input_reshaped, (0, 2, 1, 3, 4))
        self.outputs = {'Out': np.reshape(input_transposed, (-1, c, h, w))}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
