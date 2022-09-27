# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from numpy.lib.stride_tricks import as_strided
import paddle
import unittest

from op_test import OpTest


def frame_from_librosa(x, frame_length, hop_length, axis=-1):
    if axis == -1 and not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    elif axis == 0 and not x.flags["F_CONTIGUOUS"]:
        x = np.asfortranarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * x.itemsize]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * x.itemsize] + list(strides)

    else:
        raise ValueError("Frame axis={} must be either 0 or -1".format(axis))

    return as_strided(x, shape=shape, strides=strides)


class TestFrameOp(OpTest):

    def setUp(self):
        self.op_type = "frame"
        self.python_api = paddle.signal.frame
        self.shape, self.type, self.attrs = self.initTestCase()
        self.inputs = {
            'X': np.random.random(size=self.shape).astype(self.type),
        }
        self.outputs = {
            'Out': frame_from_librosa(x=self.inputs['X'], **self.attrs)
        }

    def initTestCase(self):
        input_shape = (150, )
        input_type = 'float64'
        attrs = {
            'frame_length': 50,
            'hop_length': 15,
            'axis': -1,
        }
        return input_shape, input_type, attrs

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_eager=True)
        paddle.disable_static()

    def test_check_grad_normal(self):
        paddle.enable_static()
        self.check_grad(['X'], 'Out', check_eager=True)
        paddle.disable_static()


class TestCase1(TestFrameOp):

    def initTestCase(self):
        input_shape = (150, )
        input_type = 'float64'
        attrs = {
            'frame_length': 50,
            'hop_length': 15,
            'axis': 0,
        }
        return input_shape, input_type, attrs


class TestCase2(TestFrameOp):

    def initTestCase(self):
        input_shape = (8, 150)
        input_type = 'float64'
        attrs = {
            'frame_length': 50,
            'hop_length': 15,
            'axis': -1,
        }
        return input_shape, input_type, attrs


class TestCase3(TestFrameOp):

    def initTestCase(self):
        input_shape = (150, 8)
        input_type = 'float64'
        attrs = {
            'frame_length': 50,
            'hop_length': 15,
            'axis': 0,
        }
        return input_shape, input_type, attrs


class TestCase4(TestFrameOp):

    def initTestCase(self):
        input_shape = (4, 2, 150)
        input_type = 'float64'
        attrs = {
            'frame_length': 50,
            'hop_length': 15,
            'axis': -1,
        }
        return input_shape, input_type, attrs


class TestCase5(TestFrameOp):

    def initTestCase(self):
        input_shape = (150, 4, 2)
        input_type = 'float64'
        attrs = {
            'frame_length': 50,
            'hop_length': 15,
            'axis': 0,
        }
        return input_shape, input_type, attrs


if __name__ == '__main__':
    unittest.main()
