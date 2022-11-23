#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle
import paddle.fluid as fluid


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


class TestCropTensorOp(OpTest):

    def setUp(self):
        self.op_type = "crop_tensor"
        self.shape_by_input = False
        self.offset_by_input = False
        self.unk_dim_idx = -1
        self.attrs = {}
        self.python_api = paddle.crop
        self.initTestCase()

        if self.shape_by_input:
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float64"),
                'Shape': np.array(self.crop_shape).astype("int32")
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

        crop_shape = [val for val in self.crop_shape]
        for i in range(len(self.crop_shape)):
            if self.crop_shape[i] == -1:
                crop_shape[i] = self.x_shape[i] - self.offsets[i]
        self.outputs = {'Out': crop(self.inputs['X'], self.offsets, crop_shape)}

    def initTestCase(self):
        self.x_shape = (10, 10)
        self.crop_shape = [2, 2]
        self.offsets = [1, 2]

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class TestCase1(TestCropTensorOp):

    def initTestCase(self):
        self.x_shape = (100)
        self.crop_shape = [64]
        self.offsets = [13]


class TestCase2(TestCropTensorOp):

    def initTestCase(self):
        self.x_shape = (12, 24)
        self.crop_shape = [-1, 8]
        self.offsets = [0, 0]


class TestCase3(TestCropTensorOp):

    def initTestCase(self):
        self.x_shape = (4, 8, 16)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 5, 3]
        self.shape_by_input = True


class TestCase4(TestCropTensorOp):

    def initTestCase(self):
        self.x_shape = (8, 3, 6, 6)
        self.crop_shape = [-1, 3, -1, 4]
        self.offsets = [0, 0, 1, 0]
        self.shape_by_input = True


class TestCase5(TestCropTensorOp):

    def initTestCase(self):
        self.x_shape = (2, 4, 5, 8, 8)
        self.crop_shape = [1, 1, 2, 4, 4]
        self.offsets = [1, 0, 0, 2, 2]
        self.offset_by_input = True


class TestCase6(TestCropTensorOp):

    def initTestCase(self):
        self.x_shape = (2, 2, 4, 4, 4, 2)
        self.crop_shape = [1, 1, 4, 2, 2, 2]
        self.offsets = [0, 0, 0, 0, 0, 0]
        self.shape_by_input = True
        self.offset_by_input = True


class TestCropTensorOpTensorAttr(OpTest):

    def setUp(self):
        self.op_type = "crop_tensor"
        self.OffsetsTensor = False
        self.ShapeTensor = True
        self.attrs = {}
        self.python_api = paddle.crop
        self.initTestCase()

        if self.ShapeTensor:
            shape_tensor = []
            for index, ele in enumerate(self.crop_shape):
                shape_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float64"),
                'ShapeTensor': shape_tensor
            }
            self.attrs['shape'] = self.shape_attr

        if self.OffsetsTensor:
            offsets_tensor = []
            for index, ele in enumerate(self.offsets):
                offsets_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float64"),
                'OffsetsTensor': offsets_tensor
            }
            self.attrs['offsets'] = self.offsets_attr

        self.attrs['shape'] = self.crop_shape
        self.attrs['offsets'] = self.offsets
        crop_shape = [val for val in self.crop_shape]
        for i in range(len(self.crop_shape)):
            if self.crop_shape[i] == -1:
                crop_shape[i] = self.x_shape[i] - self.offsets[i]
        self.outputs = {'Out': crop(self.inputs['X'], self.offsets, crop_shape)}

    def initTestCase(self):
        self.x_shape = (10, 10)
        self.crop_shape = (2, 2)
        self.offsets = [1, 2]
        self.shape_attr = [0, 0]

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(["X"], "Out", check_eager=True)


class TestCropTensorOpTensorAttrCase1(TestCropTensorOpTensorAttr):

    def initTestCase(self):
        self.x_shape = (16, 8, 32)
        self.crop_shape = [-1, -1, 3]
        self.offsets = [1, 5, 3]
        self.shape_attr = [-1, -1, 3]


class TestCropTensorOpTensorAttrCase2(TestCropTensorOpTensorAttr):

    def initTestCase(self):
        self.x_shape = (4, 8, 16, 8)
        self.crop_shape = [2, 2, 3, 4]
        self.offsets = [1, 5, 3, 0]
        self.shape_attr = [0, 0, 3, 4]


class TestCropTensorOpTensorAttrCase3(TestCropTensorOpTensorAttr):

    def initTestCase(self):
        self.x_shape = (16, 8, 32)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 5, 3]
        self.offsets_attr = [-1, -1, 3]
        self.ShapeTensor = False
        self.OffsetsTensor = True


class TestCropTensorOpTensorAttrCase4(TestCropTensorOpTensorAttr):

    def initTestCase(self):
        self.x_shape = (16, 8, 32)
        self.crop_shape = [2, 2, 3]
        self.shape_attr = [0, 2, 3]
        self.offsets = [1, 5, 3]
        self.offsets_attr = [-1, -1, 3]
        self.OffsetsTensor = True


class TestCropTensorException(unittest.TestCase):

    def test_exception(self):
        input1 = fluid.data(name="input1", shape=[2, 3, 6, 6], dtype="float32")
        input2 = fluid.data(name="input2", shape=[2, 3, 6, 6], dtype="float16")
        dim = fluid.data(name='dim', shape=[1], dtype='int32')
        offset = fluid.data(name='offset', shape=[1], dtype='int32')

        def attr_shape_type():
            out = paddle.crop(input1, shape=3)

        def attr_shape_dtype():
            out = paddle.crop(input1, shape=[2, 2.0, 3, 3])

        def attr_shape_value1():
            out = paddle.crop(input1, shape=[2, -2, dim, 3])

        def attr_shape_value2():
            out = paddle.crop(input1, shape=[2, 0, dim, 3])

        def attr_offsets_type():
            out = paddle.crop(input1, shape=[2, 2, 3, 3], offsets=0)

        def attr_offsets_dtype():
            out = paddle.crop(input1,
                              shape=[2, 2, 3, 3],
                              offsets=[0, 1.0, 0, 0])

        def attr_offsets_value():
            out = paddle.crop(input1,
                              shape=[2, 2, 3, 3],
                              offsets=[0, -1, offset, 0])

        def input_dtype():
            out = paddle.crop(input2, shape=[2, 2, 3, 3])

        self.assertRaises(TypeError, attr_shape_type)
        self.assertRaises(TypeError, attr_shape_dtype)
        self.assertRaises(ValueError, attr_shape_value1)
        self.assertRaises(ValueError, attr_shape_value2)
        self.assertRaises(TypeError, attr_offsets_type)
        self.assertRaises(TypeError, attr_offsets_dtype)
        self.assertRaises(ValueError, attr_offsets_value)
        self.assertRaises(TypeError, input_dtype)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    unittest.main()
