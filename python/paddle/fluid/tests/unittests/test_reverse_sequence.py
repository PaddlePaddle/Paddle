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
from op_test import OpTest
import numpy as np


class TestReverseSequenceeBase(OpTest):
    def initParameters(self):
        pass

    def setUp(self):
        self.op_type = 'reverse_sequence'
        self.size = (3, 3, 4)
        self.dtype = 'float32'

        self.initParameters()

        self.x = np.random.random(self.size).astype(self.dtype)
        # np.random.seed(1)
        self.seq_len = np.array(
            [np.random.randint(self.size[0])
             for _ in range(self.size[1])]).astype(np.int64)
        self.y = self.get_output()
        self.inputs = {'X': self.x, 'SeqLen': self.seq_len}
        self.outputs = {'Y': self.y}

    def get_output(self):
        tmp_y = np.ndarray(self.x.shape).astype(self.dtype)
        for batch_idx in range(self.x.shape[1]):
            tmp_y[0:self.seq_len[batch_idx] + 1, batch_idx, :] = self.x[
                self.seq_len[batch_idx]::-1, batch_idx, :]

        return tmp_y

    def test_output(self):
        self.check_output(0)

    def test_grad(self):
        self.check_grad(['X'], 'Y')


class TestSequenceReserve1(TestReverseSequenceeBase):
    def initParameters(self):
        self.size = (12, 10, 5)


if __name__ == '__main__':
    unittest.main()
