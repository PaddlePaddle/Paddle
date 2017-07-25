import unittest
from op_test_util import OpTestMeta
import numpy as np


class TestSigmoidOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "sigmoid"
        self.X = np.random.random((32, 100)).astype("float32")
        self.Y = 1 / (1 + np.exp(-self.X))


if __name__ == '__main__':
    unittest.main()
