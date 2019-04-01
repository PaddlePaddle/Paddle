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


class TestPixelShuffle(OpTest):
    def setUp(self):
        self.op_type = "pixel_shuffle"
        n, c, h, w = 2, 9, 4, 4
        up_factor = 3
        shape = [n, c, h, w]
        x = np.random.random(shape).astype("float32")
        new_shape = (n, c / (up_factor * up_factor), up_factor, up_factor, h, w)
        # reshape to (num,output_channel,upscale_factor,upscale_factor,h,w)
        npresult = np.reshape(x, new_shape)
        # transpose to (num,output_channel,h,upscale_factor,w,upscale_factor)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, c / (up_factor * up_factor), h * up_factor, w * up_factor]
        npresult = np.reshape(npresult, oshape)

        self.inputs = {'X': x}
        self.outputs = {'Out': npresult}
        self.attrs = {'upscale_factor': up_factor}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
