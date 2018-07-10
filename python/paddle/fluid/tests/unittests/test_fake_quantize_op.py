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

import unittest
import numpy as np
from op_test import OpTest


class TestFakeQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize"
        self.attrs = {
            'bit_length': 8,
            'quantize_type': 'abs_max',
            'window_size': 10000
        }
        self.inputs = {
            'X': np.random.random((10, 10)).astype("float32"),
            'InScales': np.zeros(self.attrs['window_size']).astype("float32"),
            'InCurrentIter': np.zeros(1).astype("float32"),
            'InMovingScale': np.zeros(1).astype("float32")
        }
        self.scale = {
            'abs_max': np.max(np.abs(self.inputs['X'])).astype("float32")
        }
        self.outputs = {
            'Out': np.round(self.inputs['X'] / self.scale['abs_max'] * (
                (1 << (self.attrs['bit_length'] - 1)) - 1)),
            'OutScales': np.zeros(self.attrs['window_size']).astype("float32"),
            'OutMovingScale':
            np.array([self.scale['abs_max']]).astype("float32"),
            'OutCurrentIter': np.zeros(1).astype("float32")
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
