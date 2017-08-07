import unittest
from op_test_util import OpTestMeta
import numpy as np


class TestRowwiseAddOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "rowwise_add"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'b': np.random.random(84).astype("float32")
        }
        self.outputs = {'Out': np.add(self.inputs['X'], self.inputs['b'])}


if __name__ == '__main__':
    unittest.main()
