import unittest
import numpy
from op_test_util import OpTestMeta


class TestSGD(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "cross_entropy"
        self.X = numpy.random.random((2, 10)).astype("float32")
        self.label = numpy.random.random((2, 10)).astype("float32")
        self.Y = self.label * numpy.log(self.X)


if __name__ == "__main__":
    unittest.main()
