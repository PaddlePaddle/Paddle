import unittest
import numpy as np
from op_test import OpTest


class TestIncrementOpPositiveStep(OpTest):
    """Test increment op with positive step
    """

    def setUp(self):
        self.op_type = "increment"
        self.inputs = {'X': np.random.random((10, 10)).astype("float32")}
        self.attrs = {'step': 14.8}
        self.outputs = {'Out': self.inputs['X'] + self.attrs['step']}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestIncrementOpNegativeStep(OpTest):
    """Test increment op with negative step
    """

    def setUp(self):
        self.op_type = "increment"
        self.inputs = {'X': np.random.random((10, 10)).astype("float32")}
        self.attrs = {'step': -3.8}
        self.outputs = {'Out': self.inputs['X'] + self.attrs['step']}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
