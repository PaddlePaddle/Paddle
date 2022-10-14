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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle

paddle.enable_static()


class TestFillZerosLikeOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "fill_zeros_like"
        self.init_dtype()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.outputs = {'Out': np.zeros_like(self.inputs["X"])}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillZerosLikeOpBool(TestFillZerosLikeOp):

    def init_dtype(self):
        self.dtype = np.bool_


class TestFillZerosLikeOpFp16(TestFillZerosLikeOp):

    def init_dtype(self):
        self.dtype = np.float16


class TestFillZerosLikeOpFp64(TestFillZerosLikeOp):

    def init_dtype(self):
        self.dtype = np.float64


class TestFillZerosLikeOpInt32(TestFillZerosLikeOp):

    def init_dtype(self):
        self.dtype = np.int32


class TestFillZerosLikeOpInt64(TestFillZerosLikeOp):

    def init_dtype(self):
        self.dtype = np.int64


if __name__ == '__main__':
    unittest.main()
