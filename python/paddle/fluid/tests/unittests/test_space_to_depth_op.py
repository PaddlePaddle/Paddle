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
import numpy as np
import paddle.fluid as fluid
from op_test import OpTest


class TestSpaceToDepthOp(OpTest):

    @staticmethod
    def helper(in_, width, height, channel, batch, blocksize, forward, out_):
        channel_out = channel // (blocksize * blocksize)
        for b in range(batch):
            for k in range(channel):
                for j in range(height):
                    for i in range(width):
                        in_index = i + width * (j + height * (k + channel * b))
                        channel2 = k % channel_out
                        offset = k // channel_out
                        width2 = i * blocksize + offset % blocksize
                        height2 = j * blocksize + offset // blocksize
                        out_index = width2 + width * blocksize * (
                            height2 + height * blocksize *
                            (channel2 + channel_out * b))
                        if forward:
                            out_[out_index] = in_[in_index]
                        else:
                            out_[in_index] = in_[out_index]

    def setUp(self):
        self.init_data()

        self.op_type = "space_to_depth"
        self.inputs = {"X": self.x}
        self.helper(self.x_1d, self.x.shape[3], self.x.shape[2],
                    self.x.shape[1], self.x.shape[0], self.blocksize,
                    self.forward, self.out_1d)
        self.out = np.reshape(self.out_1d, self.infered_shape)
        self.attrs = {"blocksize": self.blocksize}
        self.outputs = {"Out": self.out}

    def init_data(self):
        self.ori_shape = (32, 12, 6, 6)
        self.infered_shape = (32, 48, 3, 3)
        self.one_d_len = 32 * 48 * 3 * 3

        self.blocksize = 2
        self.x = np.random.random(self.ori_shape).astype('float64')
        self.x_1d = np.reshape(self.x, self.one_d_len)
        self.out = np.zeros(self.infered_shape).astype('float64')
        self.out_1d = np.reshape(self.out, self.one_d_len)
        self.forward = 1

    def test_check_output(self):
        place = fluid.core.CUDAPlace(
            0) if fluid.core.is_compiled_with_cuda() else fluid.core.CPUPlace()
        self.check_output_with_place(place, 1e-5, None, False)

    def test_check_grad(self):
        place = fluid.core.CUDAPlace(
            0) if fluid.core.is_compiled_with_cuda() else fluid.core.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out')


class TestSpaceToDepthOpBasic(TestSpaceToDepthOp):

    def init_data(self):
        self.ori_shape = (32, 8, 6, 6)
        self.infered_shape = (32, 32, 3, 3)
        self.one_d_len = 32 * 32 * 3 * 3

        self.blocksize = 2
        self.x = np.random.random(self.ori_shape).astype('float64')
        self.x_1d = np.reshape(self.x, self.one_d_len)
        self.out = np.zeros(self.infered_shape).astype('float64')
        self.out_1d = np.reshape(self.out, self.one_d_len)
        self.forward = 1


class TestSpaceToDepthOpDoubleBasic(TestSpaceToDepthOp):

    def init_data(self):
        self.ori_shape = (32, 8, 6, 6)
        self.infered_shape = (32, 32, 3, 3)
        self.one_d_len = 32 * 32 * 3 * 3

        self.blocksize = 2
        self.x = np.random.random(self.ori_shape).astype('float64')
        self.x_1d = np.reshape(self.x, self.one_d_len)
        self.out = np.zeros(self.infered_shape).astype('float64')
        self.out_1d = np.reshape(self.out, self.one_d_len)
        self.forward = 1


class TestSpaceToDepthOpWithStride3(TestSpaceToDepthOp):

    def init_data(self):
        self.ori_shape = (32, 9, 6, 6)
        self.infered_shape = (32, 81, 2, 2)
        self.one_d_len = 32 * 81 * 2 * 2

        self.blocksize = 3
        self.x = np.random.random(self.ori_shape).astype('float64')
        self.x_1d = np.reshape(self.x, self.one_d_len)
        self.out = np.zeros(self.infered_shape).astype('float64')
        self.out_1d = np.reshape(self.out, self.one_d_len)
        self.forward = 1


class TestSpaceToDepthOpWithNotSquare(TestSpaceToDepthOp):

    def init_data(self):
        self.ori_shape = (32, 9, 9, 6)
        self.infered_shape = (32, 81, 3, 2)
        self.one_d_len = 32 * 81 * 3 * 2

        self.blocksize = 3
        self.x = np.random.random(self.ori_shape).astype('float64')
        self.x_1d = np.reshape(self.x, self.one_d_len)
        self.out = np.zeros(self.infered_shape).astype('float64')
        self.out_1d = np.reshape(self.out, self.one_d_len)
        self.forward = 1


if __name__ == '__main__':
    unittest.main()
