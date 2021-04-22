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

import op_test
import unittest
import numpy as np
import struct

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestCastOp1(op_test.OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float64')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP32),
            'out_dtype': int(core.VarDesc.VarType.FP64)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        self.check_output()

    def test_grad(self):
        self.check_grad(['X'], ['Out'])


class TestCastOp2(op_test.OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float16')}
        self.outputs = {'Out': ipt.astype('float32')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP16),
            'out_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestCastOp3(op_test.OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float16')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP32),
            'out_dtype': int(core.VarDesc.VarType.FP16)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        self.check_output(atol=1e-3)


# Test FP32->BF16
@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestCaseOp4(unittest.TestCase):
    def copy_bits_from_float_to_uint16(self, f):
        return struct.unpack('<I', struct.pack('<f', f))[0] >> 16

    def convert_fp32_to_uint16(self, x):
        new_output = []
        for _ in x:
            new_output.append(self.copy_bits_from_float_to_uint16(_))
        return new_output

    def test_api(self):
        paddle.disable_static()
        ipt = np.array([1, 2, 3]).astype('float')
        data = paddle.to_tensor(ipt)
        res = paddle.cast(data, core.VarDesc.VarType.BF16)
        exp = self.convert_fp32_to_uint16(ipt)

        self.assertTrue(np.array_equal(res.numpy(), exp))
        paddle.enable_static()


# Test BF16->FP32
@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestCaseOp5(unittest.TestCase):
    def copy_bits_from_uint16_to_float(self, f):
        return struct.unpack('<f', struct.pack('<I', f << 16))[0]

    def convert_uint16_to_float(self, x):
        new_output = []
        for _ in x:
            new_output.append(self.copy_bits_from_uint16_to_float(_))
        return new_output

    def test_api(self):
        paddle.disable_static()
        ipt = np.array([1, 2, 3]).astype('uint16')
        data = paddle.to_tensor(ipt)
        res = paddle.cast(data, core.VarDesc.VarType.FP32)
        exp = self.convert_uint16_to_float(ipt)

        self.assertTrue(np.array_equal(res.numpy(), exp))
        paddle.enable_static()


class TestCastOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of cast_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.cast, x1, 'int32')
            # The input dtype of cast_op must be bool, float16, float32, float64, int32, int64, uint8.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype='int16')
            self.assertRaises(TypeError, fluid.layers.cast, x2, 'int32')

            def test_dtype_type():
                x4 = fluid.layers.data(name='x4', shape=[4], dtype='int32')
                output = fluid.layers.cast(x=x4, dtype='int16')

            self.assertRaises(TypeError, test_dtype_type)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
