#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class TestFillConstantBatchSizeLikeWhenFirstDimIsBatchSize(OpTest):
    def setUp(self):
        self.op_type = "fill_constant_batch_size_like"
        self.inputs = {'Input': np.random.random((219, 232)).astype("float32")}
        self.attrs = {'value': 3.5, 'shape': [-1, 132, 7]}

        out = np.random.random((219, 132, 7)).astype("float32")
        out.fill(3.5)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


class TestFillConstantBatchSizeLikeWhenSecondDimIsBatchSize(OpTest):
    def setUp(self):
        self.op_type = "fill_constant_batch_size_like"
        self.inputs = {'Input': np.random.random((219, 232)).astype("float32")}
        self.attrs = {
            'value': 3.5,
            'shape': [132, -1, 7],
            'input_dim_idx': 0,
            'output_dim_idx': 1
        }

        out = np.random.random((132, 219, 7)).astype("float32")
        out.fill(3.5)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


class TestFillConstantBatchSizeLikeWithLoDTensor(OpTest):
    def setUp(self):
        self.op_type = "fill_constant_batch_size_like"
        self.inputs = {
            'Input': (np.random.random((31, 28)).astype("float32"),
                      [[9, 14, 8]])
        }
        self.attrs = {
            'value': 3.5,
            'shape': [-1, 16],
            'input_dim_idx': 0,
            'output_dim_idx': 0
        }

        out = np.random.random((3, 16)).astype("float32")
        out.fill(3.5)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
