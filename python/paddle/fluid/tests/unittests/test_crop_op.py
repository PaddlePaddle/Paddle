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
from op_test import OpTest


def crop(data, offsets, crop_shape):
    def indexOf(shape, index):
        result = []
        for dim in reversed(shape):
            result.append(index % dim)
            index = index / dim
        return result[::-1]

    result = []
    for i, value in enumerate(data.flatten()):
        index = indexOf(data.shape, i)
        selected = True
        if len(index) == len(offsets):
            for j, offset in enumerate(offsets):
                selected = selected and index[j] >= offset and index[
                    j] < crop_shape[j] + offset
            if selected:
                result.append(value)
    return np.array(result).reshape(crop_shape)


class TestCropOp(OpTest):
    def setUp(self):
        self.op_type = "crop"
        self.crop_by_input = False
        self.offset_by_input = False
        self.attrs = {}
        self.initTestCase()
        if self.crop_by_input:
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float64"),
                'Y': np.random.random(self.crop_shape).astype("float64")
            }
        else:
            self.attrs['shape'] = self.crop_shape
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float64"),
            }
        if self.offset_by_input:
            self.inputs['Offsets'] = np.array(self.offsets).astype('int32')
        else:
            self.attrs['offsets'] = self.offsets
        self.outputs = {
            'Out': crop(self.inputs['X'], self.offsets, self.crop_shape)
        }

    def initTestCase(self):
        self.x_shape = (10, 10)
        self.crop_shape = (2, 2)
        self.offsets = [1, 2]

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_eager=True)


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
        self.x_shape = (4, 8, 16)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 5, 3]
        self.crop_by_input = True


class TestCase4(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 10)
        self.crop_shape = [10, 10]
        self.offsets = [0, 0]
        self.crop_by_input = True


class TestCase5(TestCropOp):
    def initTestCase(self):
        self.x_shape = (3, 4, 10)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 0, 2]
        self.offset_by_input = True


class TestCase6(TestCropOp):
    def initTestCase(self):
        self.x_shape = (10, 9, 14)
        self.crop_shape = [3, 3, 5]
        self.offsets = [3, 5, 4]
        self.crop_by_input = True
        self.offset_by_input = True


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    unittest.main()
