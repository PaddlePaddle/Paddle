import unittest
from op_test_util import OpTestMeta
import numpy as np


class TestMulOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "mul"
        self.X = np.random.random((32, 784)).astype("float32")
        self.Y = np.random.random((784, 100)).astype("float32")
        self.Out = np.dot(self.X, self.Y)


if __name__ == '__main__':
    unittest.main()
