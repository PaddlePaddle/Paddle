import unittest
import numpy
from op_test_util import OpTestMeta


class TestSGD(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "sgd"
        self.param = numpy.random.random((342, 345)).astype("float32")
        self.grad = numpy.random.random((342, 345)).astype("float32")
        self.learning_rate = 0.1
        self.param_out = self.param - self.learning_rate * self.grad


if __name__ == "__main__":
    unittest.main()
