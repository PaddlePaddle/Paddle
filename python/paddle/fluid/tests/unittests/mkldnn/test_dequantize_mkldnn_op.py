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


class TestDeQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = 'dequantize'
        self.scale = 2.0
        self.input_size = [1, 1, 5, 5]  #Naive nChw16c
        self.data_type = 'int8'
        self.set_scale()
        self.set_data_type()

        if self.data_type == 'int8':
            input = (np.random.randint(0, 100, self.input_size) - 50
                     ).astype(self.data_type)
            output = (input * (1 / self.scale)).astype('float')
        else:
            input = (np.random.randint(0, 100,
                                       self.input_size)).astype(self.data_type)
            output = (input * (1 / self.scale)).astype('float')

        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(input)}

        self.outputs = {'Output': output}

        self.attrs = {'Scale': self.scale, }

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def set_scale(self):
        pass

    def set_data_type(OpTest):
        pass


class TestDeQuantizeOp1(TestDeQuantizeOp):
    def set_scale(self):
        self.scale = 1.5

    def set_data_type(self):
        self.data_type = 'int8'


class TestDeQuantizeOp2(TestDeQuantizeOp):
    def set_scale(self):
        self.scale = 0.8

    def set_data_type(self):
        self.data_type = 'uint8'


if __name__ == '__main__':
    unittest.main()
