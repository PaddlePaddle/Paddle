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
from op_test import OpTest, convert_uint16_to_float, convert_float_to_uint16


class TestTransferDtypeOpFp32ToFp64(OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float64')}
        self.attrs = {
            'out_dtype': int(core.VarDesc.VarType.FP64),
            'in_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        self.check_output()


class TestTransferDtypeOpFp16ToFp32(OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float16')}
        self.outputs = {'Out': ipt.astype('float32')}
        self.attrs = {
            'out_dtype': int(core.VarDesc.VarType.FP32),
            'in_dtype': int(core.VarDesc.VarType.FP16)
        }
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestTransferDtypeOpFp32ToFp16(OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float16')}
        self.attrs = {
            'out_dtype': int(core.VarDesc.VarType.FP16),
            'in_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestTransferDtypeOpBf16ToFp32(OpTest):
    def setUp(self):
        ipt = np.array(np.random.randint(10, size=[10, 10])).astype('uint16')
        self.inputs = {'X': ipt}
        self.outputs = {'Out': convert_uint16_to_float(ipt)}
        self.attrs = {
            'out_dtype': int(core.VarDesc.VarType.FP32),
            'in_dtype': int(core.VarDesc.VarType.BF16)
        }
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        self.check_output()


class TestTransferDtypeFp32ToBf16(OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10]).astype('float32')
        self.inputs = {'X': ipt}
        self.outputs = {'Out': convert_float_to_uint16(ipt)}
        self.attrs = {
            'out_dtype': int(core.VarDesc.VarType.BF16),
            'in_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.op_type = 'transfer_dtype'

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
