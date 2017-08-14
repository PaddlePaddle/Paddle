import unittest
from op_test_util import OpTestMeta
import numpy as np


class TestMulOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "mul"
        self.inputs = {
            'X': np.random.random((32, 24)).astype("float32"),
            'Y': np.random.random((24, 10)).astype("float32")
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}


if __name__ == '__main__':
    unittest.main()
