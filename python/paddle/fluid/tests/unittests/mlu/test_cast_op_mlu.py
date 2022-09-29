#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")
from op_test import OpTest

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()


class TestCastOpFp32ToFp16(OpTest):

    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float16')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP32),
            'out_dtype': int(core.VarDesc.VarType.FP16)
        }
        self.op_type = 'cast'
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFp16ToFp32(OpTest):

    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float16')}
        self.outputs = {'Out': ipt.astype('float32')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP16),
            'out_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.op_type = 'cast'
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFp32ToFp64(OpTest):

    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float64')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.FP32),
            'out_dtype': int(core.VarDesc.VarType.FP64)
        }
        self.op_type = 'cast'
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt32ToInt32(OpTest):

    def setUp(self):
        ipt = np.random.randint(1000, size=(10, 10))
        self.inputs = {'X': ipt.astype('int32')}
        self.outputs = {'Out': ipt.astype('int32')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.INT32),
            'out_dtype': int(core.VarDesc.VarType.INT32)
        }
        self.op_type = 'cast'
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt32ToFp32(OpTest):

    def setUp(self):
        ipt = np.random.randint(1000, size=[10, 10])
        self.inputs = {'X': ipt.astype('int32')}
        self.outputs = {'Out': ipt.astype('float32')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.INT32),
            'out_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.op_type = 'cast'
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt16ToFp64(OpTest):

    def setUp(self):
        ipt = np.random.randint(1000, size=[10, 10])
        self.inputs = {'X': ipt.astype('int16')}
        self.outputs = {'Out': ipt.astype('int64')}
        self.attrs = {
            'in_dtype': int(core.VarDesc.VarType.INT16),
            'out_dtype': int(core.VarDesc.VarType.INT64)
        }
        self.op_type = 'cast'
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of cast_op must be Variable.
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.MLUPlace(0))
            self.assertRaises(TypeError, fluid.layers.cast, x1, 'int32')


if __name__ == '__main__':
    unittest.main()
