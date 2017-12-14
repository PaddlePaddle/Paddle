import unittest
import numpy as np
from op_test import OpTest


class TestFillConstantOp1(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value
        '''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'value': 3.8}
        self.outputs = {'Out': np.full((123, 92), 3.8)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp2(OpTest):
    def setUp(self):
        '''Test fill_constant op with default value
        '''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92]}
        self.outputs = {'Out': np.full((123, 92), 0.0)}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
