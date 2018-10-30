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
                'fix_seed': True
                }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': (self.mask_data, self.lod)
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu("mask_lm"):
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)

    def init_test_case(self):
        self.mask_id = 0
        self.voc_size = 100000
        self.input_data = np.random.randint(1, self.voc_size, size=30).astype("int32")
        
        self.lod = [[9, 4, 11, 6]]
        
        self.mask_data = np.ones(30).astype("int32") * (-1)


class TestMaskLMOp1(OpTest):
    def setUp(self):
        self.op_type = "mask_lm"
        
        self.init_test_case()

        self.inputs = {'X': (self.input_data, self.lod)}
        self.attrs = {
                'voc_size': self.voc_size, 
                'mask_id': self.mask_id, 'masked_prob': 1.0, 
                'seed': 1, 'fix_seed': True
                }
        self.outputs = {
            'Out': (self.output, self.lod),
            'Mask': (self.mask_data, self.lod)
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu("mask_lm"):
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)

    def init_test_case(self):
        self.mask_id = 0
        self.voc_size = 100000
        self.input_data = np.array([
            41804, 33577, 91710, 56352, 47466, 27739, 
            97833, 21922, 91958, 15117, 19348, 74047,
            25609, 46964, 72075, 69982, 84402, 33889,
            36279, 37262, 51326, 81762, 80231, 29753,
            37573, 58972, 94943, 29062, 99629, 51213]).astype("int32")
        
        self.lod = [[9, 4, 11, 6]]
        
        self.mask_data = self.input_data
	
        self.output = np.array([
            41804, 33577, 0, 0, 414975, 0, 0, 0, 0, 460389,
            19348, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 757987, 0, 0, 0, 0]).astype("int32")


class TestMaskLMOp2(OpTest):
    def setUp(self):
        self.op_type = "mask_lm"
        
        self.init_test_case()

        self.inputs = {'X': (self.input_data, self.lod)}
        self.attrs = {
                'voc_size': self.voc_size, 
                'mask_id': self.mask_id, 'masked_prob': 0.3, 
                'seed': 1, 'fix_seed': True
                }
        self.outputs = {
            'Out': (self.output, self.lod),
            'Mask': (self.mask_data, self.lod)
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu("mask_lm"):
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)

    def init_test_case(self):
        self.mask_id = 0
        self.voc_size = 100000
        self.input_data = np.array([
            41804, 33577, 91710, 56352, 47466, 27739, 
            97833, 21922, 91958, 15117, 19348, 74047,
            25609, 46964, 72075, 69982, 84402, 33889,
            36279, 37262, 51326, 81762, 80231, 29753,
            37573, 58972, 94943, 29062, 99629, 51213]).astype("int32")
        
        self.lod = [[9, 4, 11, 6]]
        
        self.mask_data = np.array([
	    41804, -1, -1, 56352, -1, -1, 
            97833, -1, 91958, -1, -1, -1,
	    -1, -1, -1, -1, -1, -1, 
            36279, -1, -1, -1, -1, -1,
	    37573, -1, -1, 29062, -1, -1]).astype("int32")

	self.output = np.array([
            41804, 33577, 91710, 0, 47466, 27739, 
            0, 21922, 0, 15117, 19348, 74047,
            25609, 46964, 72075, 69982, 84402, 33889, 
            0, 37262, 51326, 81762, 80231, 29753,
            0, 58972, 94943, 0, 99629, 51213]).astype("int32")


if __name__ == '__main__':
    unittest.main()
