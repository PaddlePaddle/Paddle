import unittest
import numpy as np
from op_test import OpTest


class TestExpandOpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random(12).astype("float32")}
        self.attrs = {'expand_times': [2]}
        output = np.tile(self.inputs['X'], 2)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank2_Corner(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((12, 14)).astype("float32")}
        self.attrs = {'expand_times': [1, 1]}
        output = np.tile(self.inputs['X'], (1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank2(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((12, 14)).astype("float32")}
        self.attrs = {'expand_times': [2, 3]}
        output = np.tile(self.inputs['X'], (2, 3))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank3_Corner(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((2, 4, 5)).astype("float32")}
        self.attrs = {'expand_times': [1, 1, 1]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank3(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((2, 4, 5)).astype("float32")}
        self.attrs = {'expand_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank4(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((2, 4, 5, 7)).astype("float32")}
        self.attrs = {'expand_times': [3, 2, 1, 2]}
        output = np.tile(self.inputs['X'], (3, 2, 1, 2))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
