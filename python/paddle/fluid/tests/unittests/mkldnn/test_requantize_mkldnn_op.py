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
from mkldnn_op_test import format_reorder


class TestReQuantizeOp(OpTest):
    def setUp(self):
        self.op_type = 'requantize'
        self.scale_in = 2.0
        self.scale_out = 1.5
        self.input_size = [1, 1, 5, 5]
        self.data_type = 'int8'
        self.set_scale()
        self.set_data_type()

        scale_shift = self.scale_out / self.scale_in

        if self.data_type == 'int8':
            input = (np.random.randint(0, 100, self.input_size) - 50
                     ).astype(self.data_type)
            output_tmp = np.round(input.astype('float32') *
                                  scale_shift).astype('int8')
        else:
            input = (np.random.randint(0, 100,
                                       self.input_size)).astype(self.data_type)
            output_tmp = np.round(input.astype('float32') *
                                  scale_shift).astype('uint8')

        output = format_reorder(output_tmp, self.input_size)

        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(input)}

        self.outputs = {'Output': output}

        self.attrs = {'Scale_in': self.scale_in, 'Scale_out': self.scale_out}

    def test_check_output(self):
        self.check_output()

    def set_scale(self):
        pass

    def set_data_type(OpTest):
        pass


#--------------------test requantize with s8 input--------------------


class TestReQuantizeOp1(TestReQuantizeOp):
    def set_scale(self):
        self.scale_in = 1.5
        self.scale_out = 1.5


class TestReQuantizeOp2(TestReQuantizeOp):
    def set_scale(self):
        self.scale_in = 0.1
        self.scale_out = 0.2


#--------------------test requantize with u8 input--------------------


class TestReQuantizeOp3(TestReQuantizeOp1):
    def set_data_type(self):
        self.data_type = 'uint8'


class TestReQuantizeOp4(TestReQuantizeOp2):
    def set_data_type(self):
        self.data_type = 'uint8'


if __name__ == '__main__':
    unittest.main()
