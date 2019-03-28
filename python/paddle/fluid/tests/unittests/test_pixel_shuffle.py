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
import sys
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor


class TestPixelShuffle(unittest.TestCase):
    def test_pixel_shuffle(self):
        val = np.random.random((1, 9, 4, 4)).astype('float32')
        input = layers.create_tensor(
            dtype="float32", persistable=True, name="input")
        layers.assign(input=val, output=input)
        output = layers.pixel_shuffle(input, 3)
        cpu = core.CPUPlace()
        exe = Executor(cpu)
        result = exe.run(fluid.default_main_program(),
                         feed={"input": input},
                         fetch_list=[output])
        # reshape to (num,output_channel,upscale_factor,upscale_factor,h,w)
        npresult = np.reshape(val, (1, 1, 3, 3, 4, 4))
        # transpose to (num,output_channel,h,upscale_factor,w,upscale_factor)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        npresult = np.reshape(npresult, (1, 1, 12, 12))
        self.assertTrue(np.isclose(npresult, np.array(result)).all())


if __name__ == '__main__':
    unittest.main()
