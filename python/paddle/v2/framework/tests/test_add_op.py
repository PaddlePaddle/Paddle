import unittest
import numpy as np
from op_test import OpTest


class TestAddOp(OpTest):
    def setUp(self):
        self.op_type = "add"
        self.inputs = {
            'X': np.random.random((102, 105)).astype("float32"),
            'Y': np.random.random((102, 105)).astype("float32")
        }
        self.outputs = {'Out': self.inputs['X'] + self.inputs['Y']}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
