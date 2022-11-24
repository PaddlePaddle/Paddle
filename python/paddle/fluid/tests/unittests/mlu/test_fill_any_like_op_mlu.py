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

import sys

sys.path.append("..")

import paddle
import unittest
import numpy as np
from op_test import OpTest

paddle.enable_static()


class TestFillAnyLikeOp(OpTest):

    def setUp(self):
        self.init_dtype()
        self.set_mlu()
        self.op_type = "fill_any_like"
        self.set_value()
        self.set_input()
        self.attrs = {'value': self.value}
        self.outputs = {'Out': self.value * np.ones_like(self.inputs["X"])}

    def init_dtype(self):
        self.dtype = np.float32

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)
        self.__class__.no_need_check_grad = True

    def set_input(self):
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}

    def set_value(self):
        self.value = 0.0

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillAnyLikeOp2(TestFillAnyLikeOp):

    def set_value(self):
        self.value = -0.0


class TestFillAnyLikeOp3(TestFillAnyLikeOp):

    def set_value(self):
        self.value = 1.0


class TestFillAnyLikeOp4(TestFillAnyLikeOp):

    def set_value(self):
        self.value = 1e-9


class TestFillAnyLikeOp5(TestFillAnyLikeOp):

    def set_value(self):
        if self.dtype == "float16":
            self.value = 0.05
        else:
            self.value = 5.0


class TestFillAnyLikeOpInt32(TestFillAnyLikeOp):

    def init_dtype(self):
        self.dtype = np.int32

    def set_value(self):
        self.value = -1


class TestFillAnyLikeOpInt64(TestFillAnyLikeOp):

    def init_dtype(self):
        self.dtype = np.int64

    def set_value(self):
        self.value = -1


class TestFillAnyLikeOpFloat32(TestFillAnyLikeOp):

    def init_dtype(self):
        self.dtype = np.float32

    def set_value(self):
        self.value = 0.09


if __name__ == "__main__":
    unittest.main()
