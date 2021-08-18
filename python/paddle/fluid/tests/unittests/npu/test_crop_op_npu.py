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
from test_crop_op import crop

paddle.enable_static()
np.random.seed(10)


class TestCropOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "crop"
        self.attrs = {}
        self.offset_by_input = False
        self.crop_by_input = False
        self.dtype = np.float32
        self.initTestCase()
        if self.crop_by_input:
            self.inputs = {
                'X': np.random.random(self.x_shape).astype(self.dtype),
                'Y': np.random.random(self.crop_shape).astype(self.dtype)
            }
        else:
            self.attrs['shape'] = self.crop_shape
            self.inputs = {
                'X': np.random.random(self.x_shape).astype(self.dtype),
            }

        if self.offset_by_input:
            self.inputs['Offsets'] = np.array(self.offsets).astype('int32')
        else:
            self.attrs['offsets'] = self.offsets

        if len(self.offsets) == 0:
            self.offsets = np.zeros_like(self.crop_shape)

        self.outputs = {
            'Out': crop(self.inputs['X'], self.offsets, self.crop_shape)
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def initTestCase(self):
        self.x_shape = (10, 10)
        self.crop_shape = [2, 2]
        self.offsets = [1, 2]

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestCase1(TestCropOp):
    def initTestCase(self):
        self.x_shape = (16, 8, 32)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 5, 3]


class TestCase2(TestCropOp):
    def initTestCase(self):
        self.x_shape = (15, 8)
        self.crop_shape = [15, 8]
        self.offsets = [0, 0]


class TestCase3(TestCropOp):
    def initTestCase(self):
        self.x_shape = (4, 10)
        self.crop_shape = [2, 3]
        self.offsets = [0, 2]
        self.offset_by_input = True


class TestCase4(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 9, 14)
        self.crop_shape = [3, 3, 5]
        self.offsets = []


class TestCase5(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 9, 14)
        self.crop_shape = [3, 3, 5]
        self.offsets = [3, 5, 4]
        self.offset_by_input = True


class TestCase6(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 9, 14)
        self.crop_shape = [3, 3, 5]
        self.offsets = [3, 5, 4]
        self.offset_by_input = True
        self.__class__.no_need_check_grad = True
        self.dtype = np.float16


class TestCase7(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 9, 14)
        self.crop_shape = [3, 3, 5]
        self.offsets = [3, 5, 4]
        self.offset_by_input = True
        self.dtype = np.int32


class TestCase8(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 9, 14)
        self.crop_shape = [3, 3, 5]
        self.offsets = []
        self.offset_by_input = True


class TestCase9(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 9, 14)
        self.crop_shape = [3, 3, 5]
        self.offsets = [3, 5, 4]
        self.crop_by_input = True


class TestCase10(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 9, 14)
        self.crop_shape = [3, 3, 5]
        self.offsets = [3, 5, 4]
        self.crop_by_input = True
        self.offset_by_input = True


if __name__ == '__main__':
    unittest.main()
