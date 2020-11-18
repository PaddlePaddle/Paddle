#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
from paddle import enable_static


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestElementwiseAddBf16MklDNNOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.use_mkldnn = True
        self.mkldnn_data_type = "bfloat16"
        self.axis = -1

        self.generate_data()
        self.inputs = {
            'X': convert_float_to_uint16(self.x),
            'Y': convert_float_to_uint16(self.y)
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': convert_float_to_uint16(self.out)}

    def generate_data(self):
        self.x = np.random.random(100, ).astype(np.float32)
        self.y = np.random.random(100, ).astype(np.float32)
        self.out = np.add(self.x, self.y)

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


if __name__ == '__main__':
    enable_static()
    unittest.main()
