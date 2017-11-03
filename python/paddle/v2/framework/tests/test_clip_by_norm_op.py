import unittest
import numpy as np
from op_test import OpTest


class TestClipByNormOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.initTestCase()
        input = np.random.random(self.shape).astype("float32")
        input[np.abs(input) < self.max_relative_error] = 0.5
        self.op_type = "clip_by_norm"
        self.inputs = {'X': input, }
        self.attrs = {}
        self.attrs['max_norm'] = self.max_norm
        norm = np.sqrt(np.sum(np.square(input)))
        if norm > self.max_norm:
            output = self.max_norm * input / norm
        else:
            output = input
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1.0


class TestCase1(TestClipByNormOp):
    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1e20


class TestCase2(TestClipByNormOp):
    def initTestCase(self):
        self.shape = (16, 16)
        self.max_norm = 0.1


class TestCase3(TestClipByNormOp):
    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max_norm = 1.0


if __name__ == '__main__':
    unittest.main()
