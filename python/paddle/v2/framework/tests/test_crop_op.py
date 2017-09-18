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
        self.attrs = {}
        self.initTestCase()
        self.attrs['offsets'] = self.offsets
        if self.crop_by_input:
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float32"),
                'Y': np.random.random(self.crop_shape).astype("float32")
            }
        else:
            self.attrs['shape'] = self.crop_shape
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float32"),
            }
        self.outputs = {
            'Out': crop(self.inputs['X'], self.offsets, self.crop_shape)
        }

    def initTestCase(self):
        self.x_shape = (8, 8)
        self.crop_shape = (2, 2)
        self.offsets = [1, 2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.006)


class TestCase1(TestCropOp):
    def initTestCase(self):
        self.x_shape = (16, 8, 32)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 5, 3]


class TestCase2(TestCropOp):
    def initTestCase(self):
        self.x_shape = (4, 8)
        self.crop_shape = [4, 8]
        self.offsets = [0, 0]


class TestCase3(TestCropOp):
    def initTestCase(self):
        self.x_shape = (4, 8, 16)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 5, 3]
        self.crop_by_input = True


class TestCase4(TestCropOp):
    def initTestCase(self):
        self.x_shape = (4, 4)
        self.crop_shape = [4, 4]
        self.offsets = [0, 0]
        self.crop_by_input = True


if __name__ == '__main__':
    unittest.main()
