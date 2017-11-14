import unittest
import numpy as np
from op_test import OpTest


class TestClipOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.initTestCase()
        input = np.random.random(self.shape).astype("float32")
        input[np.abs(input - self.min) < self.max_relative_error] = 0.5
        input[np.abs(input - self.max) < self.max_relative_error] = 0.5
        self.op_type = "clip"
        self.inputs = {'X': input, }
        self.attrs = {}
        self.attrs['min'] = self.min
        self.attrs['max'] = self.max
        self.outputs = {
            'Out': np.clip(self.inputs['X'], self.attrs['min'],
                           self.attrs['max'])
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=self.max_relative_error)

    def initTestCase(self):
        self.shape = (4, 4)
        self.max = 0.7
        self.min = 0.1


class TestCase1(TestClipOp):
    def initTestCase(self):
        self.shape = (8, 16, 8)
        self.max = 0.7
        self.min = 0.0


class TestCase2(TestClipOp):
    def initTestCase(self):
        self.shape = (8, 16)
        self.max = 1.0
        self.min = 0.0


class TestCase3(TestClipOp):
    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max = 0.7
        self.min = 0.2


if __name__ == '__main__':
    unittest.main()
