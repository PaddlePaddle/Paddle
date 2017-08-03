import unittest
from op_test_util import OpTestMeta
import numpy


class TestFillZerosLikeOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "fill_zeros_like"
        self.Src = numpy.random.random((219, 232)).astype("float32")
        self.Dst = numpy.zeros_like(self.Src)


if __name__ == '__main__':
    unittest.main()
