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
import six
import numpy as np
from op_test import OpTest


class TestSequenceUnpadOp(OpTest):
    def init(self):
        self.length = [2, 3, 4]
        self.x_shape = (3, 5)
        self.dtype = "float32"

    def compute(self):
        assert len(self.length) == self.x_shape[0]
        x = np.random.random(self.x_shape).astype(self.dtype)
        out_lod = [self.length]

        out = x[0, 0:self.length[0]]
        for i in six.moves.xrange(1, x.shape[0]):
            out = np.append(out, x[i, 0:self.length[i]], axis=0)

        out_shape = (sum(self.length), )
        if len(self.x_shape) == 2:
            out_shape = out_shape + (1, )
        else:
            out_shape = out_shape + self.x_shape[2:]

        self.inputs = {
            'X': x,
            'Length': np.array(self.length).astype('int64').reshape(-1, 1)
        }
        self.outputs = {'Out': (out.reshape(out_shape), out_lod)}

    def setUp(self):
        self.op_type = 'sequence_unpad'
        self.init()
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestSequenceUnpadOp2(TestSequenceUnpadOp):
    def init(self):
        self.length = [2, 3, 4]
        self.x_shape = (3, 5, 4, 3)
        self.dtype = "float32"


class TestSequenceUnpadOp3(TestSequenceUnpadOp):
    def init(self):
        self.length = [5, 2, 3, 4]
        self.x_shape = (4, 5, 3, 3, 6)
        self.dtype = "float64"


class TestSequenceUnpadOp4(TestSequenceUnpadOp):
    def init(self):
        self.length = [5, 0, 0, 4]
        self.x_shape = (4, 5, 3, 3, 6)
        self.dtype = "float64"


class TestSequenceUnpadOp4(TestSequenceUnpadOp):
    def init(self):
        self.length = [0, 4, 3, 0]
        self.x_shape = (4, 5, 3, 3, 6)
        self.dtype = "float64"


if __name__ == '__main__':
    unittest.main()
