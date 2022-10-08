# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
from paddle import enable_static


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestTransposeOp(OpTest):

    def setUp(self):
        self.op_type = "transpose2"
        self.use_mkldnn = True
        self.mkldnn_data_type = "bfloat16"
        self.init_test_case()
        self.init_test_data()
        self.axis = (0, 2, 3, 1)

        self.inputs = {'X': self.input_data}

        self.attrs = {
            'axis': list(self.axis),
            'use_mkldnn': self.use_mkldnn,
            'mkldnn_data_type': self.mkldnn_data_type
        }

        self.outputs = {
            'XShape': np.random.random(self.shape).astype(np.uint16),
            'Out': self.inputs['X'].transpose(self.axis)
        }

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), no_check_set=['XShape'])

    def init_test_case(self):
        self.shape = (2, 3, 4, 5)

    def init_test_data(self):
        self.input_data = convert_float_to_uint16(
            np.random.random(self.shape).astype(np.float32))


class TestBF16Case(TestTransposeOp):

    def init_test_case(self):
        self.shape = (2, 4, 6, 8)


if __name__ == '__main__':
    enable_static()
    unittest.main()
