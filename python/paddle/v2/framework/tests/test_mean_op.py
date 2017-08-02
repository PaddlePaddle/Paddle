import unittest
from op_test_util import OpTestMeta
import numpy as np


class TestMeanOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "mean"
        self.X = np.random.random((32, 784)).astype("float32")
        self.Out = np.mean(self.X)


if __name__ == '__main__':
    unittest.main()
