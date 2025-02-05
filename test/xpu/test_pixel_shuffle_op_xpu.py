#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def pixel_shuffle_np(x, up_factor, data_format="NCHW"):
    if data_format == "NCHW":
        n, c, h, w = x.shape
        new_shape = (
            n,
            c // (up_factor * up_factor),
            up_factor,
            up_factor,
            h,
            w,
        )
        # reshape to (num,output_channel,upscale_factor,upscale_factor,h,w)
        npresult = np.reshape(x, new_shape)
        # transpose to (num,output_channel,h,upscale_factor,w,upscale_factor)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, c // (up_factor * up_factor), h * up_factor, w * up_factor]
        npresult = np.reshape(npresult, oshape)
        return npresult
    else:
        n, h, w, c = x.shape
        new_shape = (
            n,
            h,
            w,
            c // (up_factor * up_factor),
            up_factor,
            up_factor,
        )
        # reshape to (num,h,w,output_channel,upscale_factor,upscale_factor)
        npresult = np.reshape(x, new_shape)
        # transpose to (num,h,upscale_factor,w,upscale_factor,output_channel)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, h * up_factor, w * up_factor, c // (up_factor * up_factor)]
        npresult = np.reshape(npresult, oshape)
        return npresult


class XPUTestPixelShuffleOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "pixel_shuffle"
        self.use_dynamic_create_class = False

    class TestPixelShuffleOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "pixel_shuffle"
            self.init_dtype()

            # override
            self.init_input_shape()
            self.init_attr()

            self.x = np.random.random(self.x_shape).astype(self.dtype)
            self.y = pixel_shuffle_np(
                self.x, self.attrs["upscale_factor"], self.attrs["data_format"]
            )

            self.inputs = {'X': self.x}
            self.outputs = {'Out': self.y}

        def init_input_shape(self):
            self.x_shape = [2, 64, 26, 26]

        def init_attr(self):
            self.attrs = {'upscale_factor': 2, 'data_format': "NCHW"}

        def set_xpu(self):
            self.__class__.no_need_check_grad = False
            self.place = paddle.XPUPlace(0)

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestNHWC(TestPixelShuffleOp):
        def init_input_shape(self):
            self.x_shape = [2, 64, 26, 24]

        def init_attr(self):
            self.attrs = {'upscale_factor': 2, 'data_format': "NHWC"}

    class TestUpFactor3(TestPixelShuffleOp):
        def init_input_shape(self):
            self.x_shape = [2, 27, 5, 5]

        def init_attr(self):
            self.attrs = {'upscale_factor': 3, 'data_format': "NCHW"}

    class TestUpFactor3NHWC(TestPixelShuffleOp):
        def init_input_shape(self):
            self.x_shape = [2, 27, 5, 9]

        def init_attr(self):
            self.attrs = {'upscale_factor': 3, 'data_format': "NHWC"}


support_types = get_xpu_op_support_types("pixel_shuffle")
for stype in support_types:
    create_test_class(globals(), XPUTestPixelShuffleOp, stype)

if __name__ == "__main__":
    unittest.main()
