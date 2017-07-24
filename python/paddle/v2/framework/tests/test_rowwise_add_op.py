import unittest
from op_test_util import OpTestMeta
import numpy as np


class TestRowwiseAddOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "rowwise_add"
        self.X = np.random.random((32, 784)).astype("float32")
        self.b = np.random.random(784).astype("float32")
        self.Out = np.add(self.X, self.b)


if __name__ == '__main__':
    unittest.main()
