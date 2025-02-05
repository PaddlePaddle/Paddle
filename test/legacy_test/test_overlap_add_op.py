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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


def overlap_add(x, hop_length, axis=-1):
    assert axis in [0, -1], 'axis should be 0/-1.'
    assert len(x.shape) >= 2, 'Input dims should be >= 2.'

    squeeze_output = False
    if len(x.shape) == 2:
        squeeze_output = True
        dim = 0 if axis == -1 else -1
        x = np.expand_dims(x, dim)  # batch

    n_frames = x.shape[axis]
    frame_length = x.shape[1] if axis == 0 else x.shape[-2]

    # Assure no gaps between frames.
    assert (
        0 < hop_length <= frame_length
    ), f'hop_length should be in (0, frame_length({frame_length})], but got {hop_length}.'

    seq_length = (n_frames - 1) * hop_length + frame_length

    reshape_output = False
    if len(x.shape) > 3:
        reshape_output = True
        if axis == 0:
            target_shape = [seq_length, *x.shape[2:]]
            x = x.reshape(n_frames, frame_length, np.prod(x.shape[2:]))
        else:
            target_shape = [*x.shape[:-2], seq_length]
            x = x.reshape(np.prod(x.shape[:-2]), frame_length, n_frames)

    if axis == 0:
        x = x.transpose((2, 1, 0))

    y = np.zeros(shape=[np.prod(x.shape[:-2]), seq_length], dtype=x.dtype)
    for i in range(x.shape[0]):
        for frame in range(x.shape[-1]):
            sample = frame * hop_length
            y[i, sample : sample + frame_length] += x[i, :, frame]

    if axis == 0:
        y = y.transpose((1, 0))

    if reshape_output:
        y = y.reshape(target_shape)

    if squeeze_output:
        y = y.squeeze(-1) if axis == 0 else y.squeeze(0)

    return y


class TestOverlapAddOp(OpTest):
    def setUp(self):
        self.op_type = "overlap_add"
        self.python_api = paddle.signal.overlap_add
        self.shape, self.type, self.attrs = self.initTestCase()
        self.inputs = {
            'X': np.random.random(size=self.shape).astype(self.type),
        }
        self.outputs = {'Out': overlap_add(x=self.inputs['X'], **self.attrs)}

    def initTestCase(self):
        input_shape = (50, 3)
        input_type = 'float64'
        attrs = {
            'hop_length': 4,
            'axis': -1,
        }
        return input_shape, input_type, attrs

    def test_check_output(self):
        paddle.enable_static()
        self.check_output()
        paddle.disable_static()

    def test_check_grad_normal(self):
        paddle.enable_static()
        self.check_grad(['X'], 'Out')
        paddle.disable_static()


class TestOverlapAddFP16Op(TestOverlapAddOp):
    def initTestCase(self):
        input_shape = (50, 3)
        input_type = 'float16'
        attrs = {
            'hop_length': 4,
            'axis': -1,
        }
        return input_shape, input_type, attrs


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestOverlapAddBF16Op(OpTest):
    def setUp(self):
        self.op_type = "overlap_add"
        self.python_api = paddle.signal.overlap_add
        self.shape, self.type, self.attrs = self.initTestCase()
        self.np_dtype = np.float32
        self.dtype = np.uint16
        self.inputs = {
            'X': np.random.random(size=self.shape).astype(self.np_dtype),
        }
        self.outputs = {'Out': overlap_add(x=self.inputs['X'], **self.attrs)}

        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def initTestCase(self):
        input_shape = (50, 3)
        input_type = np.uint16
        attrs = {
            'hop_length': 4,
            'axis': -1,
        }
        return input_shape, input_type, attrs

    def test_check_output(self):
        paddle.enable_static()
        self.check_output_with_place(self.place)
        paddle.disable_static()

    def test_check_grad_normal(self):
        paddle.enable_static()
        self.check_grad_with_place(self.place, ['X'], 'Out')
        paddle.disable_static()


class TestCase1(TestOverlapAddOp):
    def initTestCase(self):
        input_shape = (3, 50)
        input_type = 'float64'
        attrs = {
            'hop_length': 4,
            'axis': 0,
        }
        return input_shape, input_type, attrs


class TestCase2(TestOverlapAddOp):
    def initTestCase(self):
        input_shape = (2, 40, 5)
        input_type = 'float64'
        attrs = {
            'hop_length': 10,
            'axis': -1,
        }
        return input_shape, input_type, attrs


class TestCase3(TestOverlapAddOp):
    def initTestCase(self):
        input_shape = (5, 40, 2)
        input_type = 'float64'
        attrs = {
            'hop_length': 10,
            'axis': 0,
        }
        return input_shape, input_type, attrs


class TestCase4(TestOverlapAddOp):
    def initTestCase(self):
        input_shape = (3, 5, 12, 8)
        input_type = 'float64'
        attrs = {
            'hop_length': 5,
            'axis': -1,
        }
        return input_shape, input_type, attrs


class TestCase5(TestOverlapAddOp):
    def initTestCase(self):
        input_shape = (8, 12, 5, 3)
        input_type = 'float64'
        attrs = {
            'hop_length': 5,
            'axis': 0,
        }
        return input_shape, input_type, attrs


if __name__ == '__main__':
    unittest.main()
