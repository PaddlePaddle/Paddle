import unittest
from op_test_util import OpTestMeta
import numpy as np


class TestMeanOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "mean"
        self.inputs = {'X': np.random.random((32, 84)).astype("float32")}
        self.outputs = {'Out': np.mean(self.inputs['X'])}


if __name__ == '__main__':
    unittest.main()
