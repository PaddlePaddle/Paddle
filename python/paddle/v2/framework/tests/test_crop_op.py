import unittest
import numpy as np
from paddle.v2.framework.op import Operator
from gradient_checker import GradientChecker
from op_test_util import OpTestMeta


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


class TCropOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.type = "crop"
        self.inputs = {'X': np.random.random(self.shape).astype("float32"), }
        self.attrs = {}
        self.attrs['offsets'] = self.offsets
        self.attrs['shape'] = self.crop_shape
        self.outputs = {
            'Out': crop(self.inputs['X'], self.offsets, self.crop_shape)
        }
        print "input=%s" % self.inputs['X']

    def initTestCase(self):
        self.shape = (8, 8, 8)
        self.crop_shape = [2, 2, 2]
        self.offsets = [0, 0, 0]


#class TCase1(TCropOp):
#    def initTestCase(self):
#        self.shape = (16, 16, 16)
#        self.crop_shape = [2, 2, 3]
#        self.offsets = [1, 5, 3]

#class TCropGradOp(GradientChecker):

#    def initTestCase(self):
#        self.shape = (4, 4)
#        self.crop_shape = [2, 2]
#        self.offsets = [0, 0]

#    def setUp(self):
#        self.initTestCase()
#        self.op = Operator(
#            type="crop", X="X", Out="Out", offsets=self.offsets, shape=self.crop_shape)
#        self.inputs = {'X': np.random.random(self.shape).astype("float32"), }
#
#    def test_normal(self):
#        self.check_grad(
#            self.op, self.inputs, set(["X"]), "Out", max_relative_error=0.5)

#def test_cpu_gpu_compare(self):
#    self.compare_grad(self.op, self.inputs)

#class TestGradCase1(TestCropGradOp):

#    def initTestCase(self):
#        self.shape = (16, 16)
#        self.crop_shape = [8, 8]
#        self.offsets = [1, 1]

if __name__ == '__main__':
    unittest.main()
