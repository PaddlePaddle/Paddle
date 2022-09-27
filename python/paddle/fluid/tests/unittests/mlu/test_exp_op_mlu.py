#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestExp(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "exp"
        self.place = paddle.MLUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.rand(20, 5).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')


class TestExpFp16(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "exp"
        self.place = paddle.MLUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.rand(20, 5).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestExpNeg(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "exp"
        self.place = paddle.MLUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.random([20, 5]).astype(self.dtype)
        x -= 1
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == '__main__':
    unittest.main()
