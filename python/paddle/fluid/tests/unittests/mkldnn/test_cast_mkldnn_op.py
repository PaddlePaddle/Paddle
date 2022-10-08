#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestCastBF16ToFP32MKLDNNOp(OpTest):

    def init_data(self):
        self.out = np.random.random(size=[10, 10]).astype("float32")
        self.x = convert_float_to_uint16(self.out)

    def setUp(self):
        self.init_data()
        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out}
        prepare_dtype = lambda x: int(core.VarDesc.VarType.BF16 if x.dtype != np
                                      .float32 else core.VarDesc.VarType.FP32)
        self.attrs = {
            'in_dtype': prepare_dtype(self.x),
            'out_dtype': prepare_dtype(self.out),
            'use_mkldnn': True
        }
        self.op_type = 'cast'

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            core.CPUPlace(), ["X"],
            "Out",
            check_dygraph=False,
            user_defined_grads=[self.inputs['X']],
            user_defined_grad_outputs=[self.outputs['Out']])


class TestCastFP32ToBF16MKLDNNOp(TestCastBF16ToFP32MKLDNNOp):

    def init_data(self):
        self.x = np.random.random(size=[2, 6]).astype("float32")
        self.out = convert_float_to_uint16(self.x)


class TestCastBF16ToBF16MKLDNNOp(TestCastBF16ToFP32MKLDNNOp):

    def init_data(self):
        self.x = np.random.random(size=[6, 13]).astype("uint16")
        self.out = self.x


class TestCastFP32ToFP32MKLDNNOp(TestCastBF16ToFP32MKLDNNOp):

    def init_data(self):
        self.x = np.random.random(size=[7, 15]).astype("float32")
        self.out = self.x


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
