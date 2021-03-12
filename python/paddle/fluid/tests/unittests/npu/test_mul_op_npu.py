#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestMul(OpTest):
    def config(self):
        self.x_shape = (32, 5)
        self.y_shape = (5, 100)

    def setUp(self):
        self.set_npu()
        self.op_type = "mul"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype)
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False, atol=1e-5)


    #
class TestMulFP16(TestMatMul):
    """
    case 2
    """

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
