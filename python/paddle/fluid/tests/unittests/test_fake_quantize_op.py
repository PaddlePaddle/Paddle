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


class TestFakeQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_abs_max"
        self.attrs = {'bit_length': 8}
        self.inputs = {'X': np.random.random((124, 240)).astype("float32"), }
        scale = np.max(np.abs(self.inputs['X'])).astype("float32")
        self.outputs = {
            'Out': np.round(self.inputs['X'] / scale * (
                (1 << (self.attrs['bit_length'] - 1)) - 1)),
            'OutScale': np.array(scale).astype("float32"),
        }

    def test_check_output(self):
        self.check_output()


class TestFakeQuantizeRangeAbsMaxOp(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_range_abs_max"
        self.attrs = {
            'bit_length': int(5),
            'window_size': int(1),
            'is_test': False
        }
        x = (np.random.random((8, 16, 7, 7)) - 0.5) * 10
        x = x.astype("float32")
        self.inputs = {
            'X': x,
            'Iter': np.zeros(1).astype("int64"),
            'InScale': np.zeros(1).astype("float32")
        }
        scale = np.max(np.abs(self.inputs['X'])).astype("float32")
        out_scales = np.zeros(self.attrs['window_size']).astype("float32")
        out_scales[0] = scale
        self.outputs = {
            'Out': np.round(self.inputs['X'] / scale * (
                (1 << (self.attrs['bit_length'] - 1)) - 1)),
            'OutScale': scale,
            'OutScales': out_scales,
        }

    def test_check_output(self):
        self.check_output()


class TestFakeQuantizeRangeAbsMaxOp2(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_range_abs_max"
        self.attrs = {
            'bit_length': int(8),
            'window_size': int(1),
            'is_test': True
        }
        x = (np.random.random((8, 16, 7, 7)) - 0.5) * 10
        x = x.astype("float32")
        scale = np.max(np.abs(x)).astype("float32") - 1.0
        out_scales = np.zeros(self.attrs['window_size']).astype("float32")
        out_scales[0] = scale

        self.inputs = {
            'X': x,
            'Iter': np.zeros(1).astype("int64"),
            'InScale': scale.astype("float32")
        }
        xs = np.clip(x, -scale, scale)
        qs = np.round(xs / scale * ((1 << (self.attrs['bit_length'] - 1)) - 1))
        self.outputs = {
            'Out': qs,
            'OutScale': scale.astype("float32"),
            'OutScales': out_scales,
        }

    def test_check_output(self):
        self.check_output(no_check_set=set(['OutScale', 'OutScales']))


if __name__ == "__main__":
    unittest.main()
