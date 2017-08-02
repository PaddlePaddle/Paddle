import unittest
from op_test_util import OpTestMeta
import numpy as np


class TestSigmoidOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "sigmoid"
        self.X = np.random.random((32, 100)).astype("float32")
        self.Y = 1 / (1 + np.exp(-self.X))


#class TestSigmoidGradOp(unittest.TestCase):
#    __metaclass__ = OpTestMeta
#
#    def setUp(self):
#        self.type = "sigmoid_grad"
#        self.Y = np.random.random((32, 100)).astype("float32")
#        self.dY = np.random.random((32, 100)).astype("float32")
#        self.dX = self.dY * self.Y * (1 - self.Y)
#        print self.dX
#

if __name__ == '__main__':
    unittest.main()
