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

from __future__ import print_function
import unittest
import numpy as np
import paddle.fluid as fluid
from op_test import OpTest


class TestReorgOp(OpTest):
    @staticmethod
    def helper(in_, width, height, channel, batch, stride, forward, out_):
        channel_out = channel / (stride * stride)
        for b in range(batch):
            for k in range(channel):
                for j in range(height):
                    for i in range(width):
                        in_index = i + width * (j + height * (k + channel * b))
                        channel2 = k % channel_out
                        offset = k / channel_out
                        width2 = i * stride + offset % stride
                        height2 = j * stride + offset / stride
                        out_index = width2 + width * stride * (
                            height2 + height * stride *
                            (channel2 + channel_out * b))
                        if forward:
                            out_[out_index] = in_[in_index]
                        else:
                            out_[in_index] = in_[out_index]

    def setUp(self):
        self.init_data()

        self.op_type = "reorg"
        self.inputs = {"X": self.x}
        self.helper(self.x_1d, self.x.shape[3], self.x.shape[2],
                    self.x.shape[1], self.x.shape[0], self.stride, self.forward,
                    self.out_1d)
        self.out = np.reshape(self.out_1d, self.infered_shape)
        self.attrs = {"stride": long(self.stride)}
        self.outputs = {"Out": self.out}

    def init_data(self):
        self.ori_shape = (32, 12, 6, 6)
        self.infered_shape = (32, 48, 3, 3)
        self.one_d_len = 32 * 48 * 3 * 3

        self.stride = 2
        self.x = np.random.random(self.ori_shape).astype('float32')
        self.x_1d = np.reshape(self.x, self.one_d_len)
        self.out = np.zeros(self.infered_shape).astype('float32')
        self.out_1d = np.reshape(self.out, self.one_d_len)
        self.forward = 1

    def test_check_output(self):
        place = fluid.core.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.core.CPUPlace()
        self.check_output_with_place(place, 1e-5, None, False)

    def test_check_grad(self):
        place = fluid.core.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.core.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out')


class TestReorgOp2(TestReorgOp):
    def init_data(self):
        self.ori_shape = (32, 9, 6, 6)
        self.infered_shape = (32, 81, 2, 2)
        self.one_d_len = 32 * 81 * 2 * 2

        self.stride = 3
        self.x = np.random.random(self.ori_shape).astype('float32')
        self.x_1d = np.reshape(self.x, self.one_d_len)
        self.out = np.zeros(self.infered_shape).astype('float32')
        self.out_1d = np.reshape(self.out, self.one_d_len)
        self.forward = 1


if __name__ == '__main__':
    unittest.main()
