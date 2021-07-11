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

paddle.enable_static()

SEED = 2021
EPOCH = 100


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.inputs = {'X': self.x, 'Seed': self.seed}
        self.outputs = {'Out': self.inputs['X'], 'Mask': np.ones((32, 64)).astype('uint8')}
        self.attrs = {
            'dropout_prob': self.dropout_prob,
            'fix_seed': True,
            'is_test': False
        }

    def config(self):
        self.x = np.random.random((32, 64)).astype(self.dtype)
        self.seed = np.asarray([125], dtype="int32")
        self.dropout_prob = 0.0

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def _test_check_grad_normal(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ['X'], 'Out', check_dygraph=False)

"""
class TestSliceOp2(TestSliceOp):
    def config(self):
        self.input = np.random.random([10, 5, 6]).astype(self.dtype)
        self.starts = [0]
        self.ends = [1]
        self.axes = [1]
        self.infer_flags = [1]
        self.out = self.input[:, 0:1, :]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSliceOpFp16(TestSliceOp):
    def init_dtype(self):
        self.dtype = np.float16

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.NPUPlace(0)
"""


if __name__ == '__main__':
    unittest.main()

