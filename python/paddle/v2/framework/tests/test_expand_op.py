import unittest
import numpy as np
from op_test import OpTest


class TestExpandOpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random(12).astype("float32")}
        self.attrs = {'expandTimes': [2]}
        output = np.tile(self.inputs['X'], 2)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank2(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((12, 14)).astype("float32")}
        self.attrs = {'expandTimes': [1, 1]}
        output = np.tile(self.inputs['X'], (1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank3(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((2, 4, 5)).astype("float32")}
        self.attrs = {'expandTimes': [1, 1, 1]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank4(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((2, 4, 5, 7)).astype("float32")}
        self.attrs = {'expandTimes': [3, 2, 1, 2]}
        output = np.tile(self.inputs['X'], (3, 2, 1, 2))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
