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
import paddle.fluid.core as core


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


class TestFakeChannelWiseQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = "fake_channel_wise_quantize_abs_max"
        self.attrs = {'bit_length': 8}
        self.inputs = {
            'X': np.random.random((4, 3, 64, 64)).astype("float32"),
        }
        scales = []
        for i in range(self.inputs['X'].shape[0]):
            scales.append(np.max(np.abs(self.inputs['X'][i])).astype("float32"))
        outputs = self.inputs['X'].copy()
        for i, scale in enumerate(scales):
            outputs[i] = np.round(outputs[i] / scale * (
                (1 << (self.attrs['bit_length'] - 1)) - 1))

        self.outputs = {
            'Out': outputs,
            'OutScale': np.array(scales).astype("float32"),
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


class TestFakeQuantizeMovingOp(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_moving_average_abs_max"
        self.attrs = {
            'bit_length': int(5),
            'moving_rate': float(0.9),
            'is_test': False
        }
        accum = np.zeros(1).astype("float32")
        accum[0] = 1
        state = np.zeros(1).astype("float32")
        state[0] = 1
        scale = np.zeros(1).astype("float32")
        scale[0] = 0.001
        self.inputs = {
            'X': np.random.random((8, 16, 7, 7)).astype("float32"),
            'InScale': scale,
            'InAccum': accum,
            'InState': state,
        }

        out_accum = np.zeros(1).astype("float32")
        out_state = np.zeros(1).astype("float32")
        out_scale = np.zeros(1).astype("float32")
        out_accum[0] = self.attrs['moving_rate'] * accum[0] + np.max(
            np.abs(self.inputs['X'])).astype("float32")
        out_state[0] = self.attrs['moving_rate'] * state[0] + 1
        out_scale = out_accum / out_state
        self.outputs = {
            'Out': np.round(self.inputs['X'] / out_scale * (
                (1 << (self.attrs['bit_length'] - 1)) - 1)),
            'OutAccum': out_accum,
            'OutState': out_state,
            'OutScale': out_scale,
        }

    def test_check_output(self):
        self.check_output()


class TestMovingAverageAbsMaxScaleOp(OpTest):
    def setUp(self):
        self.op_type = "moving_average_abs_max_scale"
        self.attrs = {'moving_rate': float(0.9), 'is_test': False}
        accum = np.zeros(1).astype("float32")
        accum[0] = 1
        state = np.zeros(1).astype("float32")
        state[0] = 1
        scale = np.zeros(1).astype("float32")
        scale[0] = 0.001
        self.inputs = {
            'X': np.random.random((8, 16, 7, 7)).astype("float32"),
            'InAccum': accum,
            'InState': state,
        }

        out_accum = np.zeros(1).astype("float32")
        out_state = np.zeros(1).astype("float32")
        out_scale = np.zeros(1).astype("float32")
        out_accum[0] = self.attrs['moving_rate'] * accum[0] + np.max(
            np.abs(self.inputs['X'])).astype("float32")
        out_state[0] = self.attrs['moving_rate'] * state[0] + 1
        out_scale = out_accum / out_state
        self.outputs = {
            'Out': self.inputs['X'],
            'OutAccum': out_accum,
            'OutState': out_state,
            'OutScale': out_scale,
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
