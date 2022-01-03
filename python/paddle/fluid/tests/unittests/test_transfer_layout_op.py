# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest


# default kNCHW
class TestTransferLayoutOpkNCHWTokNHWC(OpTest):
    def setUp(self):
        ipt = np.random.random(size=[2, 3, 10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.transpose([0, 2, 3, 1])}
        self.attrs = {
            'dst_layout': 1  # kNHWC
        }
        self.op_type = 'transfer_layout'

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
