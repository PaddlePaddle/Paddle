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
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()


class TestFillAnyLikeNPUOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "fill_any_like"
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.value = 0.0

        self.init()

        self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
        self.attrs = {'value': self.value}
        self.outputs = {'Out': np.full(self.shape, self.value, self.dtype)}

    def init(self):
        pass

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillAnyLikeNPUOpInt32(TestFillAnyLikeNPUOp):
    def init(self):
        self.dtype = np.int32
        self.value = -1


class TestFillAnyLikeNPUOpFloat32(TestFillAnyLikeNPUOp):
    def init(self):
        self.dtype = np.float32
        self.value = 0.09


class TestFillAnyLikeNPUOpFloat16(TestFillAnyLikeNPUOp):
    def init(self):
        self.dtype = np.float16
        self.value = 0.05


class TestFillAnyLikeNPUOpValue1(TestFillAnyLikeNPUOp):
    def init(self):
        self.value = 1.0


class TestFillAnyLikeNPUOpValue2(TestFillAnyLikeNPUOp):
    def init(self):
        self.value = 1e-9


class TestFillAnyLikeNPUOpShape(TestFillAnyLikeNPUOp):
    def init(self):
        self.shape = [12, 10]


if __name__ == "__main__":
    unittest.main()
