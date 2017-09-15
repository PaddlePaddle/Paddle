import unittest
import numpy as np
from op_test import OpTest


class TestScaleOp(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.inputs = {'X': np.random.random((10, 10)).astype("float32")}
        self.attrs = {'scale': -2.3}
        self.outputs = {'Out': self.inputs['X'] * self.attrs['scale']}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
