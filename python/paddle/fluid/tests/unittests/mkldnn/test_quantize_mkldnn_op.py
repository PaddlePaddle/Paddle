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
from paddle.fluid.tests.unittests.op_test import OpTest


class TestQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = 'quantize'
        self.scale = 2.0
        self.input_size = [1, 1, 5, 5]  #Naive nChw16c
        self.is_negative = False
        self.set_scale()
        self.set_is_negative()

        if self.is_negative:
            input = (100 * np.random.random_sample(self.input_size) - 50
                     ).astype('float32')
            output = np.round(input * self.scale).astype('int8')
        else:
            input = (100 *
                     np.random.random_sample(self.input_size)).astype('float32')
            output = np.round(input * self.scale).astype('uint8')

        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(input)}

        self.outputs = {'Output': output}

        self.attrs = {
            'Scale': self.scale,
            'is_negative_input': self.is_negative
        }

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def set_scale(self):
        pass

    def set_is_negative(self):
        pass


class TestQuantizeOp1(TestQuantizeOp):
    def set_scale(self):
        self.scale = 1.5

    def set_is_negative(self):
        self.is_nagative = True


class TestQuantizeOp2(TestQuantizeOp):
    def set_scale(self):
        self.scale = 0.1

    def set_is_negative(self):
        self.is_nagative = False


if __name__ == '__main__':
    unittest.main()
