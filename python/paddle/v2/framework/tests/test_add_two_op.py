import unittest
from op_test_util import OpTestMeta
import numpy


class TestAddOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "add_two"
        self.X = numpy.random.random((342, 345)).astype("float32")
        self.Y = numpy.random.random((342, 345)).astype("float32")
        self.Out = self.X + self.Y


if __name__ == '__main__':
    unittest.main()
