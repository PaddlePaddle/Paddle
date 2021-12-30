#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import math
import numpy as np
import unittest
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core


class TestFoldOp(OpTest):
    """
    This is for test on fold Op
    """

    def init_data(self):
        self.batch_size = 2
        self.input_channels = 3
        self.kernel_sizes = [2, 2]

        self.strides = [1, 1]
        self.paddings = [0, 0, 0, 0]
        self.dilations = [1, 1]

        self.output_size = [4, 5]
        self.input_len = 12

        self.input_shape = [
            self.batch_size, self.input_channels * self.kernel_sizes[0] * self.kernel_sizes[1], self.input_len
        ]
        self.output_shape = [self.batch_size, self.input_channels, self.output_size[0], self.output_size[1]]
        self.x = np.random.rand(*self.input_shape).astype(np.float64)

    def calc_fold(self):
        data_im = np.zeros(self.output_shape).astype(np.float64).reshape((self.batch_size, -1))
        data_x = self.x.reshape((self.batch_size, -1))

        filter_height = self.kernel_sizes[0]
        filter_width = self.kernel_sizes[1]

        im_height = self.output_size[0]
        im_width = self.output_size[1]
        col_height = int(
            (self.output_size[0] + 2 * self.paddings[0] - (self.dilations[0] * (self.kernel_sizes[0] - 1) + 1)) /
            self.strides[0] + 1)
        col_width = int(
            (self.output_size[1] + 2 * self.paddings[1] - (self.dilations[1] * (self.kernel_sizes[1] - 1) + 1)) /
            self.strides[1] + 1)

        channels_col = self.input_shape[1]

        for b in range(self.batch_size):
            for c in range(channels_col):
                w_offset = c % filter_width
                h_offset = (c / filter_width) % filter_height
                c_im = int(c / (filter_width * filter_height))
                for h in range(col_height):
                    h_im = int(h * self.strides[0] - self.paddings[0] + h_offset * self.dilations[0])
                    for w in range(col_width):
                        w_im = int(w * self.strides[1] - self.paddings[1] + w_offset * self.dilations[1])
                        if h_im >= 0 and h_im < im_height and w_im >= 0 and w_im < im_width:
                            data_im[b, (c_im * im_height + h_im) * im_width + w_im] += data_x[
                                b, (c * col_height + h) * col_width + w]


        self.outputs = data_im.reshape(self.output_shape)

    def set_data(self):
        self.init_data()
        self.calc_fold()

        self.inputs = {'X': self.x}
        self.attrs = {
            'output_sizes': self.output_size,
            'kernel_sizes': self.kernel_sizes,
            'paddings': self.paddings,
            'dilations': self.dilations,
            'strides': self.strides
        }

        self.outputs = {'Y': self.outputs}

    def setUp(self):
        self.op_type = 'fold'
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')


class TestFoldAPI(TestFoldOp):
    """
    This is for test on paddle.nn.Fold
    """

    def setUp(self):
        self.op_type = 'fold'
        self.set_data()
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input = fluid.dygraph.to_variable(self.inputs['X'])
                m = paddle.nn.Fold(**self.attrs)
                m.eval()
                result = m(input)
                self.assertTrue(np.allclose(result.numpy(), self.outputs['Y']))

    def test_info(self):
        str(paddle.nn.Fold(**self.attrs))


if __name__ == '__main__':
    unittest.main()
