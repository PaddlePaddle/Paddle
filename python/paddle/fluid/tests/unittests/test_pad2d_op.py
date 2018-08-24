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


def pad_nchw(input, paddings, mode, constant=0):
    in_shape = input.shape
    num = in_shape[0]
    channel = in_shape[1]
    in_height = in_shape[2]
    in_width = in_shape[3]
    out_height = in_height + paddings[0] + paddings[1]
    out_width = in_width + paddings[2] + paddings[3]
    out = np.zeros([num, channel, out_height, out_width])
    input_index = 0
    out_index = 0
    if mode == "constant":
        # constant mode 
        for n in range(num):
            for c in range(channel):
                for out_h in range(out_height):
                    for out_w in range(out_w):
                        in_h = out_h - paddings[0]
                        in_w = out_w - paddings[2]
                        out[out_index + out_h * out_width + out_w] = value if (
                            in_h < 0 or in_w < 0 or in_h >= in_height or
                            in_w >= in_width
                        ) else input[input_index + in_h * in_width + in_w]
                input_index += in_height * in_width
                out_index += out_height * out_width
    elif mode == "reflect":
        # relect mode 
        for n in range(num):
            for c in range(channel):
                for out_h in range(out_height):
                    for out_w in range(out_w):
                        in_h = out_h - paddings[0]
                        in_w = out_w - paddings[2]
                        in_h = max(in_h, -in_h)
                        in_w = max(in_w, -in_w)
                        in_h = min(in_h, 2 * in_height - in_h - 2)
                        in_w = min(in_w, 2 * in_width - in_w - 2)
                        out[out_h * out_width + out_w] = input[in_h * in_width +
                                                               in_w]
                input_index += in_height * in_width
                out_index += out_height * out_width
    else:
        # edge mode
        for n in range(num):
            for c in range(channel):
                for out_h in range(out_height):
                    for out_w in range(out_w):
                        in_h = min(in_height - 1, max(out_h - paddings[0], 0))
                        in_w = min(in_width - 1, max(out_w - paddings[2], 0))
                        out[out_h * out_width + out_w] = input[in_h * in_width +
                                                               in_w]
                input_index += in_height * in_width
                out_index += out_height * out_width
    return out


def pad_nhwc(input, paddings, mode, constant=0):
    in_shape = input.shape
    num = in_shape[0]
    channel = in_shape[1]
    in_height = in_shape[2]
    in_width = in_shape[3]
    out_height = in_height + paddings[0] + paddings[1]
    out_width = in_width + paddings[2] + paddings[3]
    out = np.zeros([num, channel, out_height, out_width])
    input_index = 0
    out_index = 0
    if mode == "constant":
        # constant mode 
        for n in range(num):
            for out_h in range(out_height):
                for out_w in range(out_w):
                    in_h = out_h - paddings[0]
                    in_w = out_w - paddings[2]
                    pad_index = (out_h * out_width + out_w) * channel
                    if in_h < 0 or in_w < 0 or in_h >= in_height or in_w >= in_width:
                        for c in range(channel):
                            out[out_index + pad_index + c] = value
                    else:
                        tmp = (in_h * in_width + in_w) * channel
                        for c in range(channel):
                            out[out_index + pad_index + c] = input[input_index +
                                                                   tmp + c]
            input_index += in_height * in_width * channel
            out_index += out_height * out_width * channel
    elif mode == "reflect":  ###########################TODO#####################
        # relect mode 
        for n in range(num):
            for c in range(channel):
                for out_h in range(out_height):
                    for out_w in range(out_w):
                        in_h = out_h - paddings[0]
                        in_w = out_w - paddings[2]
                        in_h = max(in_h, -in_h)
                        in_w = max(in_w, -in_w)
                        in_h = min(in_h, 2 * in_height - in_h - 2)
                        in_w = min(in_w, 2 * in_width - in_w - 2)
                        out[out_h * out_width + out_w] = input[in_h * in_width +
                                                               in_w]
                input_index += in_height * in_width
                out_index += out_height * out_width
    else:
        # edge mode
        for n in range(num):
            for c in range(channel):
                for out_h in range(out_height):
                    for out_w in range(out_w):
                        in_h = min(in_height - 1, max(out_h - paddings[0], 0))
                        in_w = min(in_width - 1, max(out_w - paddings[2], 0))
                        out[out_h * out_width + out_w] = input[in_h * in_width +
                                                               in_w]
                input_index += in_height * in_width
                out_index += out_height * out_width
    return out


class TestPadOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = "pad"
        self.inputs = {'X': np.random.random(self.shape).astype("float32"), }
        self.attrs = {}
        self.attrs['paddings'] = np.array(self.paddings).flatten()
        self.attrs['pad_value'] = self.pad_value
        self.outputs = {
            'Out': np.pad(self.inputs['X'],
                          self.paddings,
                          mode='constant',
                          constant_values=self.pad_value)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.006)

    def initTestCase(self):
        self.shape = (16, 16)
        self.paddings = [(0, 1), (2, 3)]
        self.pad_value = 0.0


class TestCase1(TestPadOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 4)
        self.paddings = [(0, 1), (2, 3), (2, 1), (1, 1)]
        self.pad_value = 0.5


class TestCase2(TestPadOp):
    def initTestCase(self):
        self.shape = (2, 2, 2)
        self.paddings = [(0, 0), (0, 0), (1, 2)]
        self.pad_value = 1.0


class TestCase3(TestPadOp):
    def initTestCase(self):
        self.shape = (8)
        self.paddings = [(0, 1)]
        self.pad_value = 0.9


if __name__ == '__main__':
    unittest.main()
