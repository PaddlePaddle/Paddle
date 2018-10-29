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
import paddle.fluid.core as core
from op_test import OpTest


class TestMaskLMOp(OpTest):
    def setUp(self):
        self.op_type = "mask_lm"
        
        self.init_test_case()

        self.inputs = {'X': (self.input_data, self.lod)}
        self.attrs = {
                'voc_size': self.voc_size, 
                'mask_id': self.mask_id, 'masked_prob': 0.0, 
                'fix_seed': True, 'is_test': False
                }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': (self.mask_data, self.lod)
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu("mask_lm"):
            print("haha")
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)

    def init_test_case(self):
        self.mask_id = 0
        self.voc_size = 100000
        self.input_data = np.random.randint(1, self.voc_size, size=(30, 1)).astype("int32")
        self.lod = [[9, 4, 11, 6]]
        self.mask_data = np.ones((30, 1)) * (-1)

if __name__ == '__main__':
    unittest.main()
