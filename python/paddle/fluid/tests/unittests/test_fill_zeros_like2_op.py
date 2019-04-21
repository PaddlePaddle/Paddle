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
from paddle.fluid.framework import convert_np_dtype_to_dtype_
from op_test import OpTest


class TestFillZerosLike2Op(OpTest):
    def setUp(self):
        self.test_gc = True
        self.op_type = "fill_zeros_like2"
        self.dtype = np.float32
        self.init_dtype()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.outputs = {'Out': np.zeros_like(self.inputs["X"])}
        self.attrs = {'dtype': convert_np_dtype_to_dtype_(self.dtype)}

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output()


class TestFillZerosLike2OpFp16(TestFillZerosLike2Op):
    def init_dtype(self):
        self.dtype = np.float16


class TestFillZerosLike2OpFp64(TestFillZerosLike2Op):
    def init_dtype(self):
        self.dtype = np.float64


if __name__ == "__main__":
    unittest.main()
