# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle


def pixel_shuffle_np(x, up_factor, data_format="NCHW"):
    if data_format == "NCHW":
        n, c, h, w = x.shape
        new_shape = (n, c // (up_factor * up_factor), up_factor, up_factor, h,
                     w)
        # reshape to (num,output_channel,upscale_factor,upscale_factor,h,w)
        npresult = np.reshape(x, new_shape)
        # transpose to (num,output_channel,h,upscale_factor,w,upscale_factor)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, c // (up_factor * up_factor), h * up_factor, w * up_factor]
        npresult = np.reshape(npresult, oshape)
        return npresult
    else:
        n, h, w, c = x.shape
        new_shape = (n, h, w, c // (up_factor * up_factor), up_factor,
                     up_factor)
        # reshape to (num,h,w,output_channel,upscale_factor,upscale_factor)
        npresult = np.reshape(x, new_shape)
        # transpose to (num,h,upscale_factor,w,upscale_factor,output_channel)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, h * up_factor, w * up_factor, c // (up_factor * up_factor)]
        npresult = np.reshape(npresult, oshape)
        return npresult


class TestPixelShuffle(OpTest):
    def setUp(self):
        self.op_type = "pixel_shuffle"
        self.init_data_format()
        n, c, h, w = 2, 9, 4, 4

        if self.format == "NCHW":
            shape = [n, c, h, w]
        if self.format == "NHWC":
            shape = [n, h, w, c]

        up_factor = 3

        x = np.random.random(shape).astype("float64")
        npresult = pixel_shuffle_np(x, up_factor, self.format)

        self.inputs = {'X': x}
        self.outputs = {'Out': npresult}
        self.attrs = {'upscale_factor': up_factor, "data_format": self.format}

    def init_data_format(self):
        self.format = "NCHW"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestChannelLast(TestPixelShuffle):
    def init_data_format(self):
        self.format = "NHWC"


class TestPixelShuffleDygraph(unittest.TestCase):
    def run_pixel_shuffle(self, up_factor, data_format):

        n, c, h, w = 2, 9, 4, 4

        if data_format == "NCHW":
            shape = [n, c, h, w]
        if data_format == "NHWC":
            shape = [n, h, w, c]

        x = np.random.random(shape).astype("float64")

        npresult = pixel_shuffle_np(x, up_factor, data_format)

        paddle.disable_static()
        pixel_shuffle = paddle.nn.PixelShuffle(
            up_factor, data_format=data_format)
        result = pixel_shuffle(paddle.to_tensor(x))

        self.assertTrue(np.allclose(result.numpy(), npresult))

    def test_pixel_shuffle(self):
        self.run_pixel_shuffle(3, "NCHW")

    def test_channel_last(self):
        self.run_pixel_shuffle(3, "NHWC")


if __name__ == '__main__':
    unittest.main()
